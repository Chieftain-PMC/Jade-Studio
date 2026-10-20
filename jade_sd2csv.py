#!/usr/bin/env python3
"""Convert a Jade SD card log (JADEnnnn.BIN) to CSV.

The CSV has the same layout as a Jade Studio recording:
    t_s, index, flags, ch0_code ... ch7_code, ch0_mV ... ch7_mV
plus engineering-unit columns for channels calibrated in Jade Studio
(--cal, read from ~/.jade_studio.json by default). Channels that were off
read "nan". The board's log messages and a summary go in as # comments.

t_s is the board's own DRDY time, from the first sample of the file.
mV uses the gain that was set when each sample was taken.

    python jade_sd2csv.py JADE0003.BIN                 -> JADE0003.csv
    python jade_sd2csv.py E:\\JADE*.BIN --cal           (all files, with calibration)
    python jade_sd2csv.py JADE0003.BIN --info          (summary only)
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jade  # noqa: E402

NCH = 8
FS_MV = jade.FULL_SCALE_V * 1e3


def packets(data: bytes, stats: dict):
    """Payloads of every good packet in the file, in order."""
    for raw in data.split(b"\x00"):
        if not raw:
            continue
        try:
            yield jade.unframe(raw)
        except ValueError:
            stats["bad_packets"] += 1       # torn tail after a power cut, or a corrupt sector


def load_cal(path):
    """{ch: (zero_mV, scale_per_mV, unit)} from Jade Studio's settings file."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    cal = {}
    for ch, c in enumerate(d.get("channels", [])[:NCH]):
        if c.get("scale_per_mV"):
            cal[ch] = (float(c.get("zero_mV", 0.0)), float(c["scale_per_mV"]), c.get("cal_unit") or "cal")
    return cal


def convert(src: Path, dst, cal: dict, info_only=False):
    data = src.read_bytes()
    stats = dict(bad_packets=0, samples=0, gaps=0, settling=0, crc_bad=0, packets=0)
    cfgs = {}               # generation -> Config
    last_cfg = None
    logs, first_t, t_wrap, last_t = [], None, 0, None
    first_idx = last_idx = None
    tel = {}

    out = None
    if not info_only:
        out = open(dst, "w", newline="")
        out.write(f"# Jade SD log {src.name}\n")

    def header(cfg):
        out.write(f"# osr {cfg.osr}  rate {cfg.data_rate():.3f} SPS  gains {cfg.gain}  "
                  f"enabled 0x{cfg.ch_enable:02X}\n")

    cols_written = False
    fmts = ["%.6f", "%d", "%d"] + ["%.0f"] * NCH + ["%.7g"] * NCH + ["%.7g"] * len(cal)

    for p in packets(data, stats):
        stats["packets"] += 1
        t = p[0]
        if t == jade.PKT_LOG:
            text = p[2:].decode(errors="replace")
            logs.append(text)
            if out:
                out.write(f"# board: {text}\n")
        elif t == jade.PKT_RESPONSE and p[2] == 0x02 and len(p) >= 4 + jade.CONFIG_SIZE:
            cfg = jade.Config.unpack(p[4:4 + jade.CONFIG_SIZE])
            cfgs[p[1]] = last_cfg = cfg
            if out:
                if not cols_written:
                    for ch in range(NCH):
                        if ch in cal:
                            out.write(f"# ch{ch}  cal {cal[ch][1]:.9g} {cal[ch][2]}/mV zero {cal[ch][0]:.9g} mV\n")
                    out.write("# flags: 1 CRC_BAD, 2 NO_STATUS, 4 SETTLING, 8 ADC_ALERT, 16 DRDY_MISS, 32 GAP\n")
                    header(cfg)
                    cols = (["t_s", "index", "flags"] + [f"ch{c}_code" for c in range(NCH)] +
                            [f"ch{c}_mV" for c in range(NCH)] + [f"ch{c}_{cal[c][2]}" for c in cal])
                    out.write(",".join(cols) + "\n")
                    cols_written = True
                else:
                    out.write("# settings changed: ")
                    header(cfg)
        elif t == jade.PKT_TELEMETRY:
            tel = jade.parse_telemetry(p[1:])
        elif t == jade.PKT_DATA:
            blk = jade.parse_data(p)
            n = len(blk.flags)
            cfg = cfgs.get(blk.gen, last_cfg)
            gains = cfg.gain if cfg else [1] * NCH
            # device time: microseconds, wraps every ~71 min
            if last_t is not None and blk.first_t_us < last_t and last_t - blk.first_t_us > 1 << 31:
                t_wrap += 1 << 32
            last_t = blk.first_t_us
            if first_t is None:
                first_t = blk.first_t_us + t_wrap
                first_idx = blk.first_index
            rate = cfg.data_rate() if cfg else 1000.0
            t_s = (blk.first_t_us + t_wrap - first_t) / 1e6 + np.arange(n) / rate
            idx = blk.first_index + np.arange(n)
            if last_idx is not None and blk.first_index != last_idx + 1:
                stats["gaps"] += 1
            last_idx = int(idx[-1])
            stats["samples"] += n
            stats["settling"] += int(np.count_nonzero(blk.flags & 0x04))
            stats["crc_bad"] += int(np.count_nonzero(blk.flags & 0x01))
            if out:
                codes = np.full((NCH, n), np.nan)
                mv = np.full((NCH, n), np.nan)
                for j, ch in enumerate(blk.channels):
                    codes[ch] = blk.codes[:, j]
                    mv[ch] = blk.codes[:, j] * (FS_MV / 2 ** 23 / gains[ch])
                cols = [t_s, idx, blk.flags.astype(float)] + list(codes) + list(mv)
                cols += [(mv[ch] - z) * sc for ch, (z, sc, _) in cal.items()]
                np.savetxt(out, np.column_stack(cols), fmt=fmts, delimiter=",")

    dur = 0.0 if first_t is None else (last_t + t_wrap - first_t) / 1e6
    summary = (f"{src.name}: {stats['samples']} samples, {dur:.1f} s, index {first_idx}..{last_idx}, "
               f"{stats['gaps']} gap(s), {stats['settling']} settling, {stats['crc_bad']} CRC-bad, "
               f"{stats['bad_packets']} unreadable packet(s)")
    if tel:
        summary += f", board counted {tel.get('missed', 0)} missed / {tel.get('crc_errors', 0)} ADC CRC errors"
    if out:
        out.write(f"# {summary}\n")
        out.close()
    return summary, stats, logs


def main():
    ap = argparse.ArgumentParser(description="Jade SD log -> CSV")
    ap.add_argument("files", nargs="+", help="JADEnnnn.BIN file(s), wildcards allowed")
    ap.add_argument("-o", "--out", help="output file (one input) or folder")
    ap.add_argument("--cal", nargs="?", const=str(Path.home() / ".jade_studio.json"),
                    help="add engineering units from Jade Studio's calibration "
                         "(default ~/.jade_studio.json, or give a settings/profile file)")
    ap.add_argument("--info", action="store_true", help="print a summary, write nothing")
    a = ap.parse_args()

    files = [Path(f) for pat in a.files for f in (glob.glob(pat) or [pat])]
    cal = load_cal(a.cal) if a.cal else {}
    if a.cal and not cal:
        print(f"note: no calibrated channel found in {a.cal}", file=sys.stderr)
    for f in files:
        if a.out and len(files) == 1 and not os.path.isdir(a.out):
            dst = Path(a.out)
        else:
            dst = (Path(a.out) if a.out else f.parent) / (f.stem + ".csv")
        summary, _, _ = convert(f, dst, cal, a.info)
        print(summary + ("" if a.info else f"  -> {dst}"))


if __name__ == "__main__":
    main()
