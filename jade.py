#!/usr/bin/env python3
"""
Jade reference client.

Small library + command line tool that talks the Jade binary protocol
(PROTOCOL.md). It is meant as a working reference for the real PC software
and as a bench tool. Only raw ADC codes come out of here; scaling to volts
or force is left to the application.

    pip install pyserial numpy

    python jade.py info
    python jade.py config
    python jade.py set gain all 64
    python jade.py set osr 2048
    python jade.py stream -t 10 -o run1.csv
    python jade.py plot                     (live plot, needs numpy + matplotlib)
    python jade.py plot -f 20               (20 Hz low-pass, needs scipy)
    python jade.py cal 0 50 --name Thrust   (calibrate ch0 with a 50 lbf load)
    python jade.py save
    python jade.py sd ls                    (files on the SD card; also: sd get, rm, rm-logs, log, drive)
"""
from __future__ import annotations

import argparse
import csv
import queue
import binascii
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # the codec part is still usable without pyserial
    serial = None

# --------------------------------------------------------------------------
# protocol constants (keep in sync with src/protocol.h)
# --------------------------------------------------------------------------
PKT_DATA, PKT_TELEMETRY, PKT_LOG, PKT_RESPONSE = 0x10, 0x11, 0x12, 0x80

CMD = dict(
    PING=0x01, GET_CONFIG=0x02, SET_CONFIG=0x03, SET_GAIN=0x04, SET_OSR=0x05,
    SET_POWER=0x06, SET_CH_ENABLE=0x07, SET_MUX=0x08, SET_PHASE=0x09,
    SET_OCAL=0x0A, SET_GCAL=0x0B, SET_GCHOP=0x0C, SET_DC_BLOCK=0x0D,
    SET_AUTOSTART=0x0E, STREAM_START=0x10, STREAM_STOP=0x11, ADC_SYNC=0x12,
    ADC_RESET=0x13, GET_TELEMETRY=0x14, GET_SAMPLE=0x15, REG_READ=0x20,
    REG_WRITE=0x21, CONFIG_SAVE=0x30, CONFIG_LOAD=0x31, CONFIG_DEFAULTS=0x32,
    CONFIG_ERASE=0x33, IDENTIFY=0x40, SD_STATUS=0x50, SD_LIST=0x51, SD_READ=0x52,
    SD_DELETE=0x53, SD_LOGGING=0x54, SD_DRIVE=0x55, REBOOT=0x7E,
)

ERRORS = {
    0x00: "OK", 0x01: "unknown command", 0x02: "bad length", 0x03: "bad argument",
    0x04: "ADC timeout", 0x05: "register verify failed", 0x06: "ADC not detected",
    0x07: "busy", 0x08: "flash error", 0x09: "unsupported", 0x0A: "bad CRC",
    0x0B: "ADC SPI error (register read CRC)", 0x0C: "SD card",
}

SD_STATES = {0: "no card", 1: "logging", 2: "card ready, not logging", 3: "USB drive (the PC has the card)",
             4: "problem"}
SD_CARD_TYPES = {0: "", 1: "SDSC v1", 2: "SDSC", 3: "SDHC/SDXC"}
SD_FS_TYPES = {0: "", 1: "FAT12", 2: "FAT16", 3: "FAT32", 4: "exFAT"}
SD_READ_MAX = 1016

FLAG_NAMES = {0x01: "CRC_BAD", 0x02: "NO_STATUS", 0x04: "SETTLING",
              0x08: "ADC_ALERT", 0x10: "DRDY_MISS", 0x20: "GAP"}

USB_VID, USB_PID = 0x2E8A, 0x000A
CH_ALL = 0xFF
FULL_SCALE_V = 1.2          # ADS131M08 internal reference, +/-FS = 1.2 V / gain


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------
def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). binascii does it in C,
    which matters at 32 kSPS: the bit-by-bit version took most of a CPU core."""
    return binascii.crc_hqx(data, crc)


def _crc16_slow(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def cobs_encode(data: bytes) -> bytes:
    out = bytearray([0])
    code_i, code = 0, 1
    for b in data:
        if b == 0:
            out[code_i] = code
            code_i, code = len(out), 1
            out.append(0)
        else:
            out.append(b)
            code += 1
            if code == 0xFF:
                out[code_i] = code
                code_i, code = len(out), 1
                out.append(0)
    out[code_i] = code
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        code = data[i]
        if code == 0:
            raise ValueError("zero byte in COBS data")
        i += 1
        if i + code - 1 > len(data):
            raise ValueError("truncated COBS block")
        out += data[i:i + code - 1]
        i += code - 1
        if code != 0xFF and i < len(data):
            out.append(0)
    return bytes(out)


def frame(payload: bytes) -> bytes:
    return cobs_encode(payload + struct.pack("<H", crc16(payload))) + b"\x00"


def unframe(raw: bytes) -> bytes:
    d = cobs_decode(raw)
    if len(d) < 3:
        raise ValueError("packet too short")
    if struct.unpack("<H", d[-2:])[0] != crc16(d[:-2]):
        raise ValueError("CRC mismatch")
    return d[:-2]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
CONFIG_FMT = "<BBHB8B8B8h8i8IBBBBBB2x"
CONFIG_SIZE = struct.calcsize(CONFIG_FMT)
POWER_NAMES = ["VLP", "LP", "HR"]
MUX_NAMES = ["input", "shorted", "test+", "test-"]


@dataclass
class Config:
    version: int = 1
    power: int = 2
    osr: int = 4096
    ch_enable: int = 0xFF
    gain: List[int] = field(default_factory=lambda: [32] * 8)
    mux: List[int] = field(default_factory=lambda: [0] * 8)
    phase: List[int] = field(default_factory=lambda: [0] * 8)
    ocal: List[int] = field(default_factory=lambda: [0] * 8)
    gcal: List[int] = field(default_factory=lambda: [0x800000] * 8)
    dc_block: int = 0
    dc_block_dis: int = 0
    gchop_en: int = 0
    gchop_delay: int = 3
    autostart: int = 0
    sd_log_off: int = 0             # Jade: 1 = don't log to the SD card (reserved byte on OPAL)

    @classmethod
    def unpack(cls, b: bytes) -> "Config":
        v = struct.unpack(CONFIG_FMT, b[:CONFIG_SIZE])
        return cls(v[0], v[1], v[2], v[3], list(v[4:12]), list(v[12:20]), list(v[20:28]),
                   list(v[28:36]), list(v[36:44]), *v[44:50])

    def pack(self) -> bytes:
        return struct.pack(CONFIG_FMT, self.version, self.power, self.osr, self.ch_enable,
                           *self.gain, *self.mux, *self.phase, *self.ocal, *self.gcal,
                           self.dc_block, self.dc_block_dis, self.gchop_en,
                           self.gchop_delay, self.autostart, self.sd_log_off)

    @property
    def channels(self) -> List[int]:
        return [ch for ch in range(8) if self.ch_enable >> ch & 1]

    def data_rate(self, clkin_hz: float = 8.192e6) -> float:
        fmod = clkin_hz / 2
        if self.gchop_en:
            return fmod / ((2 << self.gchop_delay) + 3 * self.osr + 44)
        return fmod / self.osr

    def __str__(self) -> str:
        lines = [f"power {POWER_NAMES[self.power]}  osr {self.osr} ({self.data_rate():.1f} SPS)  "
                 f"gchop {'on' if self.gchop_en else 'off'} (delay {self.gchop_delay})  "
                 f"dc_block {self.dc_block}  autostart {self.autostart}",
                 "ch  en  gain  mux      phase   ocal       gcal      dcblk"]
        for ch in range(8):
            lines.append(f"{ch}   {'y' if self.ch_enable >> ch & 1 else '-'}   {self.gain[ch]:<4}  "
                         f"{MUX_NAMES[self.mux[ch]]:<8} {self.phase[ch]:<6}  {self.ocal[ch]:<9}  "
                         f"0x{self.gcal[ch]:06X}  {'off' if self.dc_block_dis >> ch & 1 else 'on'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# decoded packets
# --------------------------------------------------------------------------
@dataclass
class DataBlock:
    gen: int
    seq: int
    first_index: int
    first_t_us: int
    channels: List[int]
    flags: np.ndarray           # uint8, one per sample
    codes: np.ndarray           # int32, shape (samples, len(channels))


def parse_data(p: bytes) -> DataBlock:
    """Vectorised: decoding stays cheap at 32 kSPS x 8 channels."""
    _, gen, seq, idx, t_us, mask, n = struct.unpack_from("<BBHIIBB", p, 0)
    chans = [ch for ch in range(8) if mask >> ch & 1]
    k = len(chans)
    rec = 1 + 3 * k
    body = np.frombuffer(p, dtype=np.uint8, count=n * rec, offset=14).reshape(n, rec)
    flags = body[:, 0].copy()
    b = body[:, 1:].reshape(n, k, 3).astype(np.int32)
    v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    v = np.where(v & 0x800000, v - 0x1000000, v).astype(np.int32)
    return DataBlock(gen, seq, idx, t_us, chans, flags, v)


TELEMETRY_FMT = "<IBBBBHHIIIIIIIIIHI"
TELEMETRY_FIELDS = ("uptime_ms adc_ok adc_running streaming gen adc_id status conversions pushed "
                    "dropped crc_errors drdy_timeouts wreg_nak max_latency_us nominal_msps "
                    "measured_msps ring_fill missed").split()


def parse_telemetry(body: bytes) -> dict:
    return dict(zip(TELEMETRY_FIELDS, struct.unpack_from(TELEMETRY_FMT, body, 0)))


class JadeError(Exception):
    def __init__(self, cmd: int, err: int, text: str = ""):
        self.cmd, self.err = cmd, err
        super().__init__(f"cmd 0x{cmd:02X}: {ERRORS.get(err, hex(err))}" + (f" ({text})" if text else ""))


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------
def find_port() -> Optional[str]:
    if serial is None:
        return None
    ports = [p for p in serial.tools.list_ports.comports() if p.vid == USB_VID and p.pid == USB_PID]
    for p in ports:                         # Linux / macOS expose the interface name
        if p.interface and "Data" in p.interface:
            return p.device
    for p in ports:                         # Windows: interface 0 has location ...x.0
        if p.location and p.location.endswith(".0"):
            return p.device
    return sorted(p.device for p in ports)[0] if ports else None


class Jade:
    def __init__(self, port: Optional[str] = None, on_log: Optional[Callable[[int, str], None]] = None,
                 ser=None):
        """`ser` lets a test pass any object with read/write/in_waiting/close."""
        if ser is None:
            if serial is None:
                raise RuntimeError("pyserial is not installed (pip install pyserial)")
            port = port or find_port()
            if not port:
                raise RuntimeError("no Jade found, use --port")
            ser = serial.Serial(port, timeout=0.05)
            ser.dtr = True                  # the firmware treats DTR as "port open"
        self.ser = ser
        self.port = port
        self.on_log = on_log or (lambda lvl, msg: print(f"[jade {'DIWE'[lvl & 3]}] {msg}", file=sys.stderr))
        self.data: "queue.Queue[DataBlock]" = queue.Queue()
        self.telemetry: dict = {}
        self.bad_packets = 0
        self.error = ""
        self.handler_errors = 0
        self.last_handler_error = ""
        self._responses: "queue.Queue[tuple]" = queue.Queue()
        self._seq = 0
        self._lock = threading.Lock()
        self._run = True
        self._rx = threading.Thread(target=self._reader, daemon=True)
        self._rx.start()

    def close(self):
        self._run = False
        self._rx.join(timeout=1)
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.command(CMD["STREAM_STOP"], timeout=0.5)
        except Exception:
            pass                            # device may be rebooting
        self.close()

    # ---- rx ---------------------------------------------------------------
    def _reader(self):
        """Never dies on a bad packet; only a lost serial port ends it
        (the reason is kept in self.error)."""
        buf = bytearray()
        while self._run:
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except Exception as e:  # noqa: BLE001 - port gone (unplugged, board reset)
                self.error = f"serial port: {e}"
                break
            if not chunk:
                continue
            buf += chunk
            start = 0
            while True:
                i = buf.find(0, start)
                if i < 0:
                    break
                raw = bytes(buf[start:i])
                start = i + 1
                if raw:
                    try:
                        self._handle(raw)
                    except Exception:  # noqa: BLE001
                        import traceback
                        self.handler_errors += 1
                        self.last_handler_error = traceback.format_exc()
            del buf[:start]
            if len(buf) > 1 << 20:          # garbage without delimiters
                buf.clear()

    def _handle(self, raw: bytes):
        try:
            p = unframe(raw)
        except ValueError:
            self.bad_packets += 1
            return
        t = p[0]
        if t == PKT_DATA:
            self.data.put(parse_data(p))
        elif t == PKT_TELEMETRY:
            self.telemetry = parse_telemetry(p[1:])
        elif t == PKT_LOG:
            self.on_log(p[1], p[2:].decode(errors="replace"))
        elif t == PKT_RESPONSE:
            self._responses.put((p[1], p[2], p[3], p[4:]))

    # ---- commands ---------------------------------------------------------
    def command(self, cmd: int, args: bytes = b"", timeout: float = 7.0) -> bytes:
        with self._lock:
            while not self._responses.empty():     # stale answers (timeouts, bad-CRC notices)
                self._responses.get_nowait()
            self._seq = (self._seq + 1) & 0x7F
            seq = self._seq
            self.ser.write(frame(bytes([cmd, seq]) + args))
            end = time.monotonic() + timeout
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no answer to cmd 0x{cmd:02X}")
                try:
                    rseq, rcmd, err, data = self._responses.get(timeout=left)
                except queue.Empty:
                    continue
                if rseq == 0xFF and rcmd == 0xFF:
                    raise JadeError(cmd, err, data.decode(errors="replace"))
                if rseq != seq:
                    continue                    # late answer to an older command
                if err != 0:
                    raise JadeError(cmd, err, data.decode(errors="replace") if len(data) != CONFIG_SIZE else "")
                return data

    def ping(self) -> dict:
        d = self.command(CMD["PING"])
        proto, ma, mi, pa = d[0:4]
        uid = d[4:12].hex().upper()
        adc_id, adc_ok, flash_ok, spi, sysclk, clkin, cfg_size, ring = struct.unpack_from("<HBBIIIHH", d, 12)
        return dict(protocol=proto, firmware=f"{ma}.{mi}.{pa}", serial=uid, adc_id=f"0x{adc_id:04X}",
                    adc_ok=bool(adc_ok), flash=bool(flash_ok), spi_hz=spi, sys_hz=sysclk,
                    clkin_hz=clkin, config_size=cfg_size, ring_size=ring)

    def get_config(self) -> Config:
        return Config.unpack(self.command(CMD["GET_CONFIG"]))

    def set_config(self, c: Config) -> Config:
        return Config.unpack(self.command(CMD["SET_CONFIG"], c.pack()))

    def set_gain(self, ch: int, gain: int) -> Config:
        return Config.unpack(self.command(CMD["SET_GAIN"], bytes([ch, gain])))

    def set_osr(self, osr: int) -> Config:
        return Config.unpack(self.command(CMD["SET_OSR"], struct.pack("<H", osr)))

    def set_channels(self, mask: int) -> Config:
        return Config.unpack(self.command(CMD["SET_CH_ENABLE"], bytes([mask])))

    def set_mux(self, ch: int, mux: int) -> Config:
        return Config.unpack(self.command(CMD["SET_MUX"], bytes([ch, mux])))

    def set_phase(self, ch: int, phase: int) -> Config:
        return Config.unpack(self.command(CMD["SET_PHASE"], struct.pack("<Bh", ch, phase)))

    def set_ocal(self, ch: int, ocal: int) -> Config:
        return Config.unpack(self.command(CMD["SET_OCAL"], struct.pack("<Bi", ch, ocal)))

    def set_gcal(self, ch: int, gcal: int) -> Config:
        return Config.unpack(self.command(CMD["SET_GCAL"], struct.pack("<BI", ch, gcal)))

    def set_gchop(self, enable: bool, delay: int = 3) -> Config:
        return Config.unpack(self.command(CMD["SET_GCHOP"], bytes([int(enable), delay])))

    def set_dc_block(self, code: int, disable_mask: int = 0) -> Config:
        return Config.unpack(self.command(CMD["SET_DC_BLOCK"], bytes([code, disable_mask])))

    def set_autostart(self, on: bool) -> Config:
        return Config.unpack(self.command(CMD["SET_AUTOSTART"], bytes([int(on)])))

    def stream_start(self):
        while not self.data.empty():
            self.data.get_nowait()
        self.command(CMD["STREAM_START"])

    def stream_stop(self):
        self.command(CMD["STREAM_STOP"])

    def sync(self):
        self.command(CMD["ADC_SYNC"])

    def reset_adc(self) -> Config:
        return Config.unpack(self.command(CMD["ADC_RESET"]))

    def get_telemetry(self) -> dict:
        return parse_telemetry(self.command(CMD["GET_TELEMETRY"]))

    def get_sample(self) -> dict:
        d = self.command(CMD["GET_SAMPLE"])
        idx, t_us, status, flags, gen, mask = struct.unpack_from("<IIHBBB", d, 0)
        codes = list(struct.unpack_from("<8i", d, 13))
        return dict(index=idx, t_us=t_us, status=status, flags=flags, gen=gen, mask=mask, codes=codes)

    def reg_read(self, addr: int, count: int = 1) -> List[int]:
        d = self.command(CMD["REG_READ"], bytes([addr, count]))
        return list(struct.unpack(f"<{count}H", d))

    def reg_write(self, addr: int, values: List[int]) -> List[int]:
        d = self.command(CMD["REG_WRITE"], bytes([addr]) + struct.pack(f"<{len(values)}H", *values))
        return list(struct.unpack(f"<{len(values)}H", d))

    def save(self) -> Config:
        return Config.unpack(self.command(CMD["CONFIG_SAVE"]))

    def load(self) -> Config:
        return Config.unpack(self.command(CMD["CONFIG_LOAD"]))

    def defaults(self) -> Config:
        return Config.unpack(self.command(CMD["CONFIG_DEFAULTS"]))

    def erase_saved(self):
        self.command(CMD["CONFIG_ERASE"])

    def identify(self):
        self.command(CMD["IDENTIFY"])

    # ---- SD card ----------------------------------------------------------
    def sd_status(self) -> dict:
        d = self.command(CMD["SD_STATUS"])
        (state, log_on, ctype, fstype, fno, written, dropped, werr, size, free, width,
         clock) = struct.unpack_from("<BBBBIQIIQQBI", d, 0)
        return dict(state=state, state_text=SD_STATES.get(state, str(state)), logging_on=bool(log_on),
                    card=SD_CARD_TYPES.get(ctype, str(ctype)), fs=SD_FS_TYPES.get(fstype, str(fstype)),
                    file=f"JADE{fno:04d}.BIN" if fno else "", bytes_written=written,
                    dropped_samples=dropped, write_errors=werr, card_bytes=size, free_bytes=free,
                    bus_width=width, clock_hz=clock, message=d[45:].decode(errors="replace"))

    def sd_list(self) -> List[dict]:
        """Files in the card's root: name, size, active (being written), log (JADEnnnn/OPALnnnn.BIN)."""
        out, start = [], 0
        while True:
            d = self.command(CMD["SD_LIST"], struct.pack("<H", start))
            total, _, count = struct.unpack_from("<HHB", d, 0)
            p = 5
            for _ in range(count):
                flags, size, nl = struct.unpack_from("<BQB", d, p)
                name = d[p + 10:p + 10 + nl].decode(errors="replace")
                out.append(dict(name=name, size=size, active=bool(flags & 1), log=bool(flags & 2)))
                p += 10 + nl
            start += count
            if start >= total or count == 0:
                return out

    def sd_download(self, name: str, dest, size: Optional[int] = None,
                    progress: Optional[Callable[[int, int], bool]] = None, window: int = 8) -> int:
        """Copy a file from the card to `dest` (a path or a binary file object).
        Several reads are kept in flight, so the USB round trip doesn't set the speed.
        `progress(done, size)` returning False cancels (the partial file stays)."""
        if size is None:
            size = next((f["size"] for f in self.sd_list() if f["name"] == name), None)
            if size is None:
                raise JadeError(CMD["SD_READ"], 0x0C, "no such file")
        own = isinstance(dest, (str, bytes)) or hasattr(dest, "__fspath__")
        f = open(dest, "wb") if own else dest
        try:
            done = 0
            while done < size:
                # the port is shared: let other commands through every ~64 kB
                n = self._sd_read_batch(name, done, min(size, done + 64 * SD_READ_MAX), f, window)
                if n == 0:
                    raise JadeError(CMD["SD_READ"], 0x0C, "file shorter than listed")
                done += n
                if progress and progress(done, size) is False:
                    break
            return done
        finally:
            if own:
                f.close()

    def _sd_read_batch(self, name: str, start: int, stop: int, f, window: int) -> int:
        nb = name.encode()
        with self._lock:
            while not self._responses.empty():
                self._responses.get_nowait()
            pending = {}                        # seq -> offset
            chunks = {}
            nxt = done = start
            short = False
            try:
                while done < stop and not short:
                    while nxt < stop and len(pending) < window:
                        self._seq = (self._seq + 1) & 0x7F
                        n = min(SD_READ_MAX, stop - nxt)
                        self.ser.write(frame(bytes([CMD["SD_READ"], self._seq]) + struct.pack("<QH", nxt, n) + nb))
                        pending[self._seq] = nxt
                        nxt += n
                    try:
                        rseq, rcmd, err, data = self._responses.get(timeout=5.0)
                    except queue.Empty:
                        raise TimeoutError("SD read: no answer")
                    if rseq == 0xFF and rcmd == 0xFF:
                        raise JadeError(CMD["SD_READ"], err, data.decode(errors="replace"))
                    if rseq not in pending:
                        continue
                    off = pending.pop(rseq)
                    if err:
                        raise JadeError(CMD["SD_READ"], err, data.decode(errors="replace"))
                    chunks[off] = data
                    while done in chunks:       # answers come in order, but don't rely on it
                        b = chunks.pop(done)
                        f.write(b)
                        done += len(b)
                        if len(b) < min(SD_READ_MAX, stop - (done - len(b))):
                            short = True        # end of the file
                            break
            finally:
                # collect what's still in flight so the next command isn't confused
                end = time.monotonic() + 2.0
                while pending and time.monotonic() < end:
                    try:
                        pending.pop(self._responses.get(timeout=0.2)[0], None)
                    except queue.Empty:
                        pass
        return done - start

    def sd_delete(self, name: str) -> int:
        d = self.command(CMD["SD_DELETE"], name.encode())
        return struct.unpack("<H", d)[0]

    def sd_delete_all_logs(self) -> int:
        """Every JADEnnnn.BIN / OPALnnnn.BIN; logging carries on in a new file."""
        return self.sd_delete("*")

    def sd_logging(self, on: bool) -> Config:
        """Log to the card (default) or not. Like autostart, `save` keeps it over power cycles."""
        return Config.unpack(self.command(CMD["SD_LOGGING"], bytes([int(on)])))

    def sd_drive(self, on: bool):
        """Hand the card to the PC as a USB drive (logging pauses), or take it back.
        Ejecting the drive on the PC also gives it back."""
        self.command(CMD["SD_DRIVE"], bytes([int(on)]))

    def reboot(self, bootloader: bool = False):
        self.command(CMD["REBOOT"], bytes([1 if bootloader else 0]))


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _ch(s: str) -> int:
    return CH_ALL if s == "all" else int(s)


def cmd_stream(dev: Jade, args):
    cfg = dev.get_config()
    rate = cfg.data_rate()
    out = open(args.output, "w", newline="") if args.output else None
    writer = csv.writer(out) if out else None
    chans = cfg.channels
    if writer:
        writer.writerow(["index", "t_us", "flags"] + [f"ch{c}" for c in chans])

    n = gaps = flagged = 0
    last_index = None
    t_end = time.monotonic() + args.time
    t_print = time.monotonic()
    dev.stream_start()
    try:
        while time.monotonic() < t_end:
            try:
                blk = dev.data.get(timeout=0.2)
            except queue.Empty:
                continue
            if blk.channels != chans:
                chans = blk.channels
                if writer:
                    writer.writerow(["# channels changed"] + [f"ch{c}" for c in chans])
            for i, (fl, row) in enumerate(zip(blk.flags, blk.codes)):
                idx = blk.first_index + i
                if last_index is not None and idx != last_index + 1:
                    gaps += 1
                last_index = idx
                if fl:
                    flagged += 1
                if writer:
                    # timestamp: DRDY time of the first sample in the packet,
                    # following samples extrapolated with the nominal rate
                    t = blk.first_t_us + round(i * 1e6 / rate)
                    writer.writerow([idx, t & 0xFFFFFFFF, int(fl)] + row.tolist())
            n += len(blk.codes)
            if time.monotonic() - t_print > 1.0 and blk.codes:
                t_print = time.monotonic()
                row = blk.codes[-1]
                vals = "  ".join(f"ch{c}:{int(v):>9}" for c, v in zip(blk.channels, row))
                print(f"{n:>9} samples  gaps {gaps}  flagged {flagged}  |  {vals}")
    finally:
        dev.stream_stop()
        if out:
            out.close()
    print(f"done: {n} samples, {gaps} index gaps, {flagged} flagged, {dev.bad_packets} bad packets")


# --------------------------------------------------------------------------
# calibration (engineering units) and live filtering - host side only
# --------------------------------------------------------------------------
def code_to_mV(code, gain):
    return code * (FULL_SCALE_V * 1e3 / 2**23 / gain)


def load_cal(path: Optional[str]) -> dict:
    """Per-channel calibration file (JSON), keys are channel numbers:

        {
          "0": {"name": "Thrust", "unit": "lbf",
                "capacity": 5000, "sensitivity_mV_V": 3.0, "excitation_V": 5.0},
          "2": {"name": "Ox tank", "unit": "lbf",
                "scale_per_mV": 333.3, "zero_mV": 0.118}
        }

    value = (mV - zero_mV) * scale_per_mV. When scale_per_mV is missing it is
    taken from the datasheet numbers: capacity / (sensitivity_mV_V * excitation_V).
    `python jade.py cal` measures zero_mV and scale_per_mV with a known load,
    which is more accurate than the datasheet (it includes the real excitation).
    """
    if not path:
        return {}
    import json
    import os
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    cal = {}
    for k, c in raw.items():
        c = dict(c)
        if "scale_per_mV" not in c:
            c["scale_per_mV"] = c["capacity"] / (c["sensitivity_mV_V"] * c["excitation_V"])
        c.setdefault("zero_mV", 0.0)
        c.setdefault("unit", "lbf")
        c.setdefault("name", f"ch{k}")
        cal[int(k)] = c
    return cal


def save_cal(path: str, ch: int, entry: dict):
    import json
    import os
    raw = {}
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
    raw[str(ch)] = entry
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


class LiveFilter:
    """Causal filter that keeps its state between chunks (so a live plot has
    no seams). lowpass = 4th order Butterworth corner in Hz (needs scipy),
    avg = moving average over N samples. NaN samples (settling) are held."""

    def __init__(self, rate: float, lowpass: Optional[float] = None, avg: int = 0):
        import numpy as np
        self.np = np
        self.sos = self.zi = None
        self.avg = max(0, int(avg))
        self.tail = None
        self.last = None
        if lowpass:
            if lowpass >= rate / 2:
                raise ValueError(f"low-pass {lowpass} Hz must be below Nyquist ({rate / 2:.0f} Hz)")
            try:
                from scipy.signal import butter
            except ImportError:
                sys.exit("--lowpass needs scipy: pip install scipy (or use --avg N)")
            self.sos = butter(4, lowpass, fs=rate, output="sos")

    def __call__(self, x):
        np = self.np
        x = np.array(x, dtype=float)
        nan = np.isnan(x)
        if nan.all() and self.last is None:
            return x
        # hold last good value over NaN so the filter state stays finite
        if nan.any():
            idx = np.where(~nan, np.arange(len(x)), -1)
            np.maximum.accumulate(idx, out=idx)
            first = self.last if self.last is not None else x[~nan][0]
            x = np.where(idx >= 0, x[np.maximum(idx, 0)], first)
        self.last = x[-1]
        y = x
        if self.sos is not None:
            from scipy.signal import sosfilt, sosfilt_zi
            if self.zi is None:
                self.zi = sosfilt_zi(self.sos) * y[0]
            y, self.zi = sosfilt(self.sos, y, zi=self.zi)
        if self.avg > 1:
            prev = self.tail if self.tail is not None else np.full(self.avg - 1, y[0])
            ext = np.concatenate([prev, y])
            c = np.cumsum(np.insert(ext, 0, 0.0))
            y = (c[self.avg:] - c[:-self.avg]) / self.avg
            self.tail = ext[-(self.avg - 1):]
        y = np.array(y, dtype=float)
        y[nan] = np.nan
        return y


def cmd_plot(dev: Jade, args):
    """Live plot. Keys: t = tare (zero every trace), r = remove tare, space = pause."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        sys.exit("plot needs numpy and matplotlib: pip install numpy matplotlib")

    cfg = dev.get_config()
    chans = cfg.channels
    if args.channels:
        want = [int(c) for c in args.channels.split(",")]
        chans = [c for c in want if c in cfg.channels]
        if not chans:
            sys.exit(f"none of {want} is enabled (enabled: {cfg.channels})")
    cal = load_cal(args.cal)

    def unit_of(ch):
        if ch in cal:
            return cal[ch]["unit"]
        return {"code": "code", "mV": "mV", "uV": "µV"}[args.units]

    def label_of(ch):
        return f"{cal[ch]['name']} (ch{ch})" if ch in cal else f"ch{ch}"

    def convert(codes, ch):
        """raw codes -> displayed unit for this channel"""
        if ch in cal:
            c = cal[ch]
            return (code_to_mV(codes, state["cfg"].gain[ch]) - c["zero_mV"]) * c["scale_per_mV"]
        if args.units == "code":
            return codes
        mv = code_to_mV(codes, state["cfg"].gain[ch])
        return mv * 1e3 if args.units == "uV" else mv

    state = dict(cfg=cfg, gen=None, paused=False, rate=cfg.data_rate())
    n = max(10, int(args.window * state["rate"]))
    buf = np.full((len(chans), n), np.nan)
    tare = np.zeros(len(chans))
    stride = max(1, n // 4000)              # keep matplotlib responsive

    def make_filters():
        return [LiveFilter(state["rate"], args.lowpass, args.avg) for _ in chans]
    filters = make_filters()

    def t_axis():
        return ((np.arange(n) - n) / state["rate"])[::stride]

    fig, axes = plt.subplots(len(chans), 1, sharex=True, squeeze=False, figsize=(11, 1.6 + 1.4 * len(chans)))
    axes = axes[:, 0]
    lines = []
    for ax, ch in zip(axes, chans):
        (ln,) = ax.plot(t_axis(), buf[0, ::stride], lw=0.8)
        ax.set_ylabel(f"{label_of(ch)}\n[{unit_of(ch)}]")
        ax.grid(True, alpha=0.3)
        lines.append(ln)
    axes[-1].set_xlabel("time [s]")
    axes[-1].set_xlim(t_axis()[0], 0)
    fig.suptitle("Jade", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    filt_txt = " ".join(x for x in (f"LP {args.lowpass:g} Hz" if args.lowpass else "",
                                   f"avg {args.avg}" if args.avg > 1 else "") if x) or "no filter"

    def on_key(ev):
        if ev.key == "t":
            tare[:] = np.nanmean(buf[:, -max(1, int(state["rate"] * 0.5)):], axis=1)
        elif ev.key == "r":
            tare[:] = 0
        elif ev.key == " ":
            state["paused"] = not state["paused"]
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        nonlocal buf, filters
        blocks = []
        while not dev.data.empty():
            blocks.append(dev.data.get_nowait())
        if state["paused"] or not blocks:
            return lines
        if blocks[-1].gen != state["gen"]:
            first = state["gen"] is None
            state["gen"] = blocks[-1].gen
            state["cfg"] = dev.get_config()      # gain / OSR may have changed
            if not first and abs(state["cfg"].data_rate() - state["rate"]) > 1e-6:
                state["rate"] = state["cfg"].data_rate()
                filters = make_filters()
                for ln in lines:
                    ln.set_xdata(t_axis())
                axes[-1].set_xlim(t_axis()[0], 0)
        cols = []
        for b in blocks:
            if not b.codes:
                continue
            arr = np.asarray(b.codes, dtype=float)                   # samples x channels in packet
            bad = np.array([bool(f & 0x05) for f in b.flags])       # SETTLING or CRC_BAD
            arr[bad, :] = np.nan
            col = np.full((len(chans), len(arr)), np.nan)
            for k, ch in enumerate(chans):
                if ch in b.channels:
                    col[k] = convert(arr[:, b.channels.index(ch)], ch)
            cols.append(col)
        if not cols:
            return lines
        new = np.concatenate(cols, axis=1)
        for k in range(len(chans)):
            new[k] = filters[k](new[k])
        buf = np.concatenate([buf, new], axis=1)[:, -n:]
        shown = buf - tare[:, None]
        for k, (ax, ln) in enumerate(zip(axes, lines)):
            ln.set_ydata(shown[k, ::stride])
            y = shown[k][~np.isnan(shown[k])]
            if y.size:
                lo, hi = float(y.min()), float(y.max())
                pad = (hi - lo) * 0.1 or max(abs(hi) * 0.01, 1e-9)
                ax.set_ylim(lo - pad, hi + pad)
                u = unit_of(chans[k])
                ax.set_title(f"{label_of(chans[k])}  last {y[-1]:.5g} {u}   mean {y.mean():.5g}   "
                             f"p-p {hi - lo:.3g}   (gain {state['cfg'].gain[chans[k]]})",
                             fontsize=9, loc="left")
        tel = dev.telemetry
        fig.suptitle(f"Jade  {state['rate']:.0f} SPS  {filt_txt}   dropped {tel.get('dropped', 0)}  "
                     f"crc {tel.get('crc_errors', 0)}  missed {tel.get('missed', 0)}"
                     f"{'   [PAUSED]' if state['paused'] else ''}   keys: t tare, r untare, space pause",
                     fontsize=9)
        return lines

    dev.stream_start()
    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # noqa: F841 (keep a reference)
    try:
        plt.show()
    finally:
        dev.stream_stop()


def cmd_cal(dev: Jade, args):
    """Two-point calibration of one channel with a known load."""
    cfg = dev.get_config()
    ch = args.channel
    if ch not in cfg.channels:
        sys.exit(f"channel {ch} is not enabled")
    rate = cfg.data_rate()

    def average_mV(seconds):
        while not dev.data.empty():
            dev.data.get_nowait()
        vals, t_end = [], time.monotonic() + seconds
        while time.monotonic() < t_end:
            try:
                b = dev.data.get(timeout=0.2)
            except queue.Empty:
                continue
            k = b.channels.index(ch)
            vals += [int(row[k]) for f, row in zip(b.flags, b.codes) if not f & 0x05]
        if not vals:
            sys.exit("no data received")
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        g = cfg.gain[ch]
        return code_to_mV(mean, g), code_to_mV(sd, g), len(vals)

    print(f"ch{ch}: gain {cfg.gain[ch]}, {rate:.0f} SPS, averaging {args.seconds:g} s per point")
    dev.stream_start()
    try:
        input("1) Remove all load from the cell, then press Enter... ")
        z, zsd, nz = average_mV(args.seconds)
        print(f"   zero  = {z:.6f} mV  (sd {zsd * 1e3:.2f} uV, {nz} samples)")
        input(f"2) Apply the known load ({args.known:g} {args.unit}), then press Enter... ")
        m, msd, nm = average_mV(args.seconds)
        print(f"   loaded = {m:.6f} mV  (sd {msd * 1e3:.2f} uV, {nm} samples)")
    finally:
        dev.stream_stop()
    if abs(m - z) < 5 * max(zsd, msd, 1e-6):
        sys.exit("signal change is within the noise, check the load / wiring / gain")
    scale = args.known / (m - z)
    entry = {"name": args.name or f"ch{ch}", "unit": args.unit, "scale_per_mV": scale, "zero_mV": z,
             "gain": cfg.gain[ch], "known_load": args.known, "date": time.strftime("%Y-%m-%d %H:%M")}
    save_cal(args.file, ch, entry)
    print(f"scale = {scale:.4f} {args.unit}/mV   ->  saved to {args.file}")
    print(f"noise at zero ~ {zsd * abs(scale):.3g} {args.unit} rms")


def cmd_sd(dev: Jade, args):
    a, v = args.action, args.args
    if a == "status":
        for k, x in dev.sd_status().items():
            print(f"{k:16}: {x}")
    elif a == "ls":
        for f in dev.sd_list():
            print(f"{f['size']:>12}  {f['name']}{'  (being written)' if f['active'] else ''}")
    elif a == "get":
        if not v:
            raise SystemExit("sd get NAME [OUT]")
        out = v[1] if len(v) > 1 else v[0]
        t0 = time.monotonic()
        n = dev.sd_download(v[0], out, progress=lambda d, s: print(f"\r{d}/{s} bytes", end="", file=sys.stderr))
        dt = max(time.monotonic() - t0, 1e-3)
        print(f"\n{out}: {n} bytes, {n / dt / 1024:.0f} kB/s", file=sys.stderr)
    elif a == "rm":
        for name in v:
            dev.sd_delete(name)
    elif a == "rm-logs":
        print(f"deleted {dev.sd_delete_all_logs()} log file(s)")
    elif a in ("log", "drive"):
        if not v or v[0] not in ("on", "off"):
            raise SystemExit(f"sd {a} on|off")
        if a == "log":
            dev.sd_logging(v[0] == "on")
        else:
            dev.sd_drive(v[0] == "on")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Jade DAQ reference client")
    ap.add_argument("-p", "--port", help="serial port of the Jade Data interface")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sub.add_parser("config")
    sub.add_parser("telemetry")
    sub.add_parser("sample")
    s = sub.add_parser("set", help="gain|mux|phase|ocal|gcal <ch|all> <v>, osr <v>, en <hexmask>, "
                                   "power <0-2>, gchop <0|1> [delay], dcblock <code> [dis_mask], autostart <0|1>")
    s.add_argument("what")
    s.add_argument("values", nargs="+")
    pl = sub.add_parser("plot", help="live plot of the stream")
    pl.add_argument("-w", "--window", type=float, default=5.0, help="seconds shown")
    pl.add_argument("-c", "--channels", help="e.g. 0,1,4 (default: all enabled)")
    pl.add_argument("-u", "--units", choices=["code", "mV", "uV"], default="mV")
    pl.add_argument("-f", "--lowpass", type=float, help="low-pass corner in Hz (4th order Butterworth, needs scipy)")
    pl.add_argument("-a", "--avg", type=int, default=0, help="moving average over N samples")
    pl.add_argument("--cal", default="cal.json", help="calibration file (default cal.json, used if present)")
    ca = sub.add_parser("cal", help="two-point calibration of one channel with a known load")
    ca.add_argument("channel", type=int)
    ca.add_argument("known", type=float, help="known load applied in step 2")
    ca.add_argument("--unit", default="lbf")
    ca.add_argument("--name", help="label shown in the plot, e.g. Thrust")
    ca.add_argument("--seconds", type=float, default=3.0)
    ca.add_argument("--file", default="cal.json")
    st = sub.add_parser("stream")
    st.add_argument("-t", "--time", type=float, default=5.0, help="seconds")
    st.add_argument("-o", "--output", help="CSV file")
    r = sub.add_parser("regs")
    r.add_argument("addr", nargs="?", default="0")
    r.add_argument("count", nargs="?", default="51")
    w = sub.add_parser("wreg")
    w.add_argument("addr")
    w.add_argument("values", nargs="+")
    for name in ("save", "load", "defaults", "erase", "reset", "sync", "identify"):
        sub.add_parser(name)
    sd = sub.add_parser("sd", help="SD card: status | ls | get NAME [OUT] | rm NAME | rm-logs | log on|off | drive on|off")
    sd.add_argument("action", choices=["status", "ls", "get", "rm", "rm-logs", "log", "drive"])
    sd.add_argument("args", nargs="*")
    rb = sub.add_parser("reboot")
    rb.add_argument("--boot", action="store_true", help="reboot into the USB bootloader")
    args = ap.parse_args(argv)

    with Jade(args.port) as dev:
        c = args.cmd
        if c == "info":
            for k, v in dev.ping().items():
                print(f"{k:12}: {v}")
        elif c == "config":
            print(dev.get_config())
        elif c == "telemetry":
            for k, v in dev.get_telemetry().items():
                print(f"{k:15}: {v}")
        elif c == "sample":
            s_ = dev.get_sample()
            cfg = dev.get_config()
            print(f"#{s_['index']}  flags 0x{s_['flags']:02X}  status 0x{s_['status']:04X}")
            for ch, code in enumerate(s_["codes"]):
                if s_["mask"] >> ch & 1:
                    uv = code * FULL_SCALE_V / 2**23 / cfg.gain[ch] * 1e6
                    print(f"  ch{ch}: {code:>9}   ~{uv:10.2f} uV (gain {cfg.gain[ch]})")
        elif c == "set":
            v = args.values
            w_ = args.what
            if w_ == "gain":
                cfg = dev.set_gain(_ch(v[0]), int(v[1]))
            elif w_ == "mux":
                cfg = dev.set_mux(_ch(v[0]), int(v[1]))
            elif w_ == "phase":
                cfg = dev.set_phase(_ch(v[0]), int(v[1]))
            elif w_ == "ocal":
                cfg = dev.set_ocal(_ch(v[0]), int(v[1], 0))
            elif w_ == "gcal":
                cfg = dev.set_gcal(_ch(v[0]), int(v[1], 0))
            elif w_ == "osr":
                cfg = dev.set_osr(int(v[0]))
            elif w_ == "en":
                cfg = dev.set_channels(int(v[0], 16))
            elif w_ == "power":
                cfg = Config.unpack(dev.command(CMD["SET_POWER"], bytes([int(v[0])])))
            elif w_ == "gchop":
                cfg = dev.set_gchop(bool(int(v[0])), int(v[1]) if len(v) > 1 else 3)
            elif w_ == "dcblock":
                cfg = dev.set_dc_block(int(v[0]), int(v[1], 16) if len(v) > 1 else 0)
            elif w_ == "autostart":
                cfg = dev.set_autostart(bool(int(v[0])))
            else:
                ap.error(f"unknown setting {w_}")
            print(cfg)
        elif c == "plot":
            cmd_plot(dev, args)
        elif c == "cal":
            cmd_cal(dev, args)
        elif c == "stream":
            cmd_stream(dev, args)
        elif c == "regs":
            a, n = int(args.addr, 0), int(args.count, 0)
            vals = dev.reg_read(a, n)
            for i, v in enumerate(vals):
                print(f"{a + i:02X}: {v:04X}", end="\n" if i % 8 == 7 else "   ")
            print()
        elif c == "wreg":
            print([f"{v:04X}" for v in dev.reg_write(int(args.addr, 0), [int(x, 16) for x in args.values])])
        elif c == "save":
            print(dev.save())
        elif c == "load":
            print(dev.load())
        elif c == "defaults":
            print(dev.defaults())
        elif c == "erase":
            dev.erase_saved()
        elif c == "reset":
            print(dev.reset_adc())
        elif c == "sync":
            dev.sync()
        elif c == "identify":
            dev.identify()
        elif c == "sd":
            cmd_sd(dev, args)
        elif c == "reboot":
            dev.reboot(args.boot)


if __name__ == "__main__":
    main()
