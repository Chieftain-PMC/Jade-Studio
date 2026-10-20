#!/usr/bin/env python3
"""
Jade Studio - live acquisition front-end for the Jade DAQ.

    pip install pyserial numpy PySide6 pyqtgraph scipy
    python jade_gui.py

Everything is live: gains, OSR, channel enables, input mux, filters,
calibration, tare, recording and markers can all be changed while the board
is streaming. The firmware only sends raw codes; every conversion
(code -> mV -> engineering unit) happens here.

Keys: Space pause/follow, T tare all, M marker, R start/stop recording.
"""
from __future__ import annotations

import faulthandler
import json
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot, QByteArray
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QAbstractItemView, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QScrollArea,
    QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar, QVBoxLayout,
    QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jade  # noqa: E402

try:
    import serial.tools.list_ports as list_ports
except ImportError:  # pragma: no cover
    list_ports = None

try:
    import scipy.signal  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

APP_NAME = "Jade Studio"
SETTINGS_FILE = Path.home() / ".jade_studio.json"
OLD_SETTINGS_FILE = Path.home() / ".opal_studio.json"   # OPAL Studio (rev1): picked up once
CRASH_LOG = Path.home() / "jade_studio_crash.log"
MAX_BACKLOG = 20000               # data packets waiting; beyond that the PC can't keep up


def crash_log(text: str):
    try:
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{text}\n")
    except OSError:
        pass
NCH = 8
GAINS = [1, 2, 4, 8, 16, 32, 64, 128]
OSRS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16256]
MUX_NAMES = ["Input", "Shorted", "Test +", "Test −"]
UNITS = ["mV", "µV", "code", "cal"]
CAL_UNITS = ["lbf", "N", "kN", "kgf", "kg", "psi", "bar", "°C", "custom"]
WINDOWS = [1, 2, 5, 10, 20, 30, 60]
COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4db6ac", "#f06292"]
FS_MV = jade.FULL_SCALE_V * 1e3
FLAG_BAD = 0x01 | 0x04            # CRC_BAD | SETTLING -> not plotted


# ==========================================================================
# helpers
# ==========================================================================
class _Bridge(QObject):
    """Runs callables on the GUI thread (signals cross threads safely)."""
    call = Signal(object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run)

    @Slot(object)
    def _run(self, fn):
        fn()


class DeviceWorker:
    """One background thread for every device command, in order, so the GUI
    never freezes while a config change waits for the ADC."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.bridge = _Bridge()
        self.busy = 0

    def run(self, fn: Callable, ok: Optional[Callable] = None, err: Optional[Callable] = None):
        self.busy += 1

        def job():
            try:
                res = fn()
            except Exception as e:  # noqa: BLE001
                exc = e
                self.bridge.call.emit(lambda: self._done(err, exc))
            else:
                self.bridge.call.emit(lambda: self._done(ok, res))

        self.pool.submit(job)

    def _done(self, cb, value):
        self.busy -= 1
        if cb:
            cb(value)


@dataclass
class ChanSettings:
    name: str
    color: str
    show: bool = True
    unit: str = "mV"
    cal_unit: str = "lbf"
    scale_per_mV: float = 0.0       # 0 = not calibrated
    zero_mV: float = 0.0
    cal_note: str = ""

    @property
    def calibrated(self) -> bool:
        return self.scale_per_mV != 0.0

    def unit_label(self) -> str:
        return self.cal_unit if self.unit == "cal" else self.unit


def to_display(s: ChanSettings, mv: np.ndarray, gain) -> np.ndarray:
    if s.unit == "mV":
        return mv
    if s.unit == "µV":
        return mv * 1e3
    if s.unit == "code":
        return mv * (2 ** 23) * gain / FS_MV
    return (mv - s.zero_mV) * s.scale_per_mV


def adc_id_hint(adc_id) -> str:
    """What an ID register read that isn't 0x28xx usually means."""
    try:
        v = int(str(adc_id), 0)
    except ValueError:
        return ""
    if v == 0x0000:
        return ("0x0000 = DOUT stays low: the ADC is not powered (check AVDD and DVDD at the ADC pins), "
                "has no 8.192 MHz clock on CLKIN (Y2, JP1 must be open), or is held in reset (/SYNC-/RESET low).")
    if v == 0xFFFF:
        return ("0xFFFF = DOUT stays high or floats: the ADC is not driving it. Check DVDD at the ADC, "
                "that /CS reaches the ADC, and the DOUT solder joint.")
    if (v >> 8) == 0x28:
        return ("The ID is right but the frame CRC failed: noise or a bad joint on SCLK/DIN/DOUT, "
                "or a supply dip on the ADC.")
    return ("Garbled read: bad solder joint or noise on SCLK/DIN/DOUT, or the ADC is only half powered "
            "(AVDD or DVDD missing).")


def fmt(v: float, digits: int = 5) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}g}"


class SampleStore:
    """Contiguous growing buffers; when full the oldest quarter is dropped."""

    def __init__(self, cap: int = 1_000_000):
        self.cap = cap
        self.t = np.empty(cap)
        self.mv = np.empty((NCH, cap), np.float32)       # raw, mV (NaN = no data)
        self.fmv = np.empty((NCH, cap), np.float32)      # filtered, mV
        self.gain = np.empty((NCH, cap), np.uint8)
        self.n = 0

    def clear(self):
        self.n = 0

    def append(self, t, mv, fmv, gain):
        k = len(t)
        if k > self.cap // 2:
            t, mv, fmv, gain = t[-self.cap // 2:], mv[:, -self.cap // 2:], fmv[:, -self.cap // 2:], gain[:, -self.cap // 2:]
            k = len(t)
        if self.n + k > self.cap:
            keep = max(0, min(self.n, self.cap * 3 // 4 - k))
            src = slice(self.n - keep, self.n)
            self.t[:keep] = self.t[src]
            self.mv[:, :keep] = self.mv[:, src]
            self.fmv[:, :keep] = self.fmv[:, src]
            self.gain[:, :keep] = self.gain[:, src]
            self.n = keep
        e = self.n + k
        self.t[self.n:e] = t
        self.mv[:, self.n:e] = mv
        self.fmv[:, self.n:e] = fmv
        self.gain[:, self.n:e] = gain
        self.n = e

    def window(self, seconds: float, t_end: Optional[float] = None) -> slice:
        if self.n == 0:
            return slice(0, 0)
        t_end = self.t[self.n - 1] if t_end is None else t_end
        a = int(np.searchsorted(self.t[:self.n], t_end - seconds))
        b = int(np.searchsorted(self.t[:self.n], t_end, side="right"))
        return slice(a, b)


class Recorder:
    """CSV recording: time, index, flags, raw code + mV for all 8 channels,
    plus engineering units for calibrated channels. Markers go in as
    comment lines so any CSV reader can skip them.

    Writing happens on its own thread so a big recording never slows the
    plots or the controls."""

    def __init__(self, path: Path, cfg: jade.Config, chans: List[ChanSettings], info: dict):
        self.path = path
        self.f = open(path, "w", newline="")
        self.rows = 0
        self.t0 = time.monotonic()
        self.cal_ch = [ch for ch in range(NCH) if chans[ch].calibrated]
        self.cal = {ch: (chans[ch].zero_mV, chans[ch].scale_per_mV) for ch in self.cal_ch}
        w = self.f.write
        w(f"# Jade recording {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        w(f"# firmware {info.get('firmware', '?')}  board {info.get('serial', '?')}\n")
        w(f"# osr {cfg.osr}  rate {cfg.data_rate():.3f} SPS  gains {cfg.gain}  enabled 0x{cfg.ch_enable:02X}\n")
        for ch in range(NCH):
            c = chans[ch]
            cal = f"  cal {c.scale_per_mV:.9g} {c.cal_unit}/mV zero {c.zero_mV:.9g} mV" if c.calibrated else ""
            w(f"# ch{ch} '{c.name}'{cal}\n")
        w("# flags: 1 CRC_BAD, 2 NO_STATUS, 4 SETTLING, 8 ADC_ALERT, 16 DRDY_MISS, 32 GAP\n")
        cols = ["t_s", "index", "flags"] + [f"ch{c}_code" for c in range(NCH)] + [f"ch{c}_mV" for c in range(NCH)]
        cols += [f"ch{c}_{chans[c].cal_unit}" for c in self.cal_ch]
        w(",".join(cols) + "\n")
        self.fmts = ["%.6f", "%d", "%d"] + ["%.0f"] * NCH + ["%.7g"] * NCH + ["%.7g"] * len(self.cal)
        self.error = ""
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def write(self, t, index, flags, codes, mv):
        """codes/mv: (8, k) arrays with NaN for channels not present."""
        self.rows += len(t)
        self._q.put(("w", (t, index, flags, codes, mv)))

    def comment(self, text: str):
        self._q.put(("c", text))

    def close(self, wait: float = 3.0) -> bool:
        """Finish writing. Returns False if it is still flushing after `wait`
        seconds (it then completes in the background)."""
        self._q.put(("stop", None))
        self._thread.join(timeout=wait)
        return not self._thread.is_alive()

    @property
    def backlog(self) -> int:
        return self._q.qsize()

    @property
    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def _run(self):
        try:
            while True:
                kind, item = self._q.get()
                if kind == "stop":
                    break
                if kind == "c":
                    self.f.write(f"# {item}\n")
                    continue
                # merge what is already waiting into one write
                chunks = [item]
                while len(chunks) < 64:
                    try:
                        nk, ni = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nk != "w":
                        self._write_chunks(chunks)
                        chunks = []
                        if nk == "stop":
                            self.f.close()
                            return
                        self.f.write(f"# {ni}\n")
                        continue
                    chunks.append(ni)
                self._write_chunks(chunks)
        except Exception:  # noqa: BLE001
            self.error = traceback.format_exc()
            crash_log("recorder:\n" + self.error)
        finally:
            try:
                self.f.close()
            except Exception:  # noqa: BLE001
                pass

    def _write_chunks(self, chunks):
        if not chunks:
            return
        t = np.concatenate([c[0] for c in chunks])
        idx = np.concatenate([c[1] for c in chunks])
        flags = np.concatenate([c[2] for c in chunks])
        codes = np.concatenate([c[3] for c in chunks], axis=1)
        mv = np.concatenate([c[4] for c in chunks], axis=1)
        cols = [t, idx, flags] + list(codes) + list(mv)
        cols += [(mv[ch] - z) * sc for ch, (z, sc) in self.cal.items()]
        np.savetxt(self.f, np.column_stack(cols), fmt=self.fmts, delimiter=",")


# ==========================================================================
# calibration dialog
# ==========================================================================
class CalDialog(QDialog):
    def __init__(self, main: "MainWindow", ch: int):
        super().__init__(main)
        self.main, self.ch = main, ch
        s = main.chan[ch]
        self.setWindowTitle(f"Calibrate ch{ch} — {s.name}")
        self.zero: Optional[float] = s.zero_mV if s.calibrated else None
        self.loaded: Optional[float] = None

        lay = QVBoxLayout(self)
        top = QFormLayout()
        self.name = QLineEdit(s.name)
        self.unit = QComboBox()
        self.unit.setEditable(True)
        self.unit.addItems(CAL_UNITS)
        self.unit.setCurrentText(s.cal_unit)
        top.addRow("Name", self.name)
        top.addRow("Unit", self.unit)
        lay.addLayout(top)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        # --- known load --------------------------------------------------
        w = QWidget()
        f = QFormLayout(w)
        self.avg_s = QDoubleSpinBox()
        self.avg_s.setRange(0.2, 30)
        self.avg_s.setValue(3.0)
        self.avg_s.setSuffix(" s")
        f.addRow("Average over", self.avg_s)
        self.btn_zero = QPushButton("1. Capture zero (no load)")
        self.lbl_zero = QLabel("—")
        f.addRow(self.btn_zero, self.lbl_zero)
        self.known = QDoubleSpinBox()
        self.known.setRange(-1e9, 1e9)
        self.known.setDecimals(4)
        self.known.setValue(50.0)
        f.addRow("2. Known load", self.known)
        self.btn_load = QPushButton("3. Capture loaded")
        self.lbl_load = QLabel("—")
        f.addRow(self.btn_load, self.lbl_load)
        self.lbl_res = QLabel("")
        self.lbl_res.setWordWrap(True)
        f.addRow(self.lbl_res)
        self.tabs.addTab(w, "Known load")

        # --- datasheet ---------------------------------------------------
        w = QWidget()
        f = QFormLayout(w)
        self.cap = QDoubleSpinBox()
        self.cap.setRange(0, 1e9)
        self.cap.setDecimals(3)
        self.cap.setValue(5000)
        self.sens = QDoubleSpinBox()
        self.sens.setRange(0.001, 100)
        self.sens.setDecimals(4)
        self.sens.setValue(3.0)
        self.sens.setSuffix(" mV/V")
        self.exc = QDoubleSpinBox()
        self.exc.setRange(0.1, 30)
        self.exc.setDecimals(3)
        self.exc.setValue(5.0)
        self.exc.setSuffix(" V")
        f.addRow("Capacity", self.cap)
        f.addRow("Sensitivity", self.sens)
        f.addRow("Excitation", self.exc)
        self.btn_zero2 = QPushButton("Capture zero (no load)")
        self.lbl_zero2 = QLabel("zero: 0 mV (uses current zero if captured)")
        f.addRow(self.btn_zero2, self.lbl_zero2)
        note = QLabel("Datasheet scaling ignores the real excitation voltage and cable losses. "
                      "A known load is more accurate.")
        note.setWordWrap(True)
        f.addRow(note)
        self.tabs.addTab(w, "Datasheet")

        # --- manual ------------------------------------------------------
        w = QWidget()
        f = QFormLayout(w)
        self.m_scale = QDoubleSpinBox()
        self.m_scale.setRange(-1e12, 1e12)
        self.m_scale.setDecimals(6)
        self.m_scale.setValue(s.scale_per_mV or 1.0)
        self.m_zero = QDoubleSpinBox()
        self.m_zero.setRange(-1e6, 1e6)
        self.m_zero.setDecimals(6)
        self.m_zero.setValue(s.zero_mV)
        self.m_zero.setSuffix(" mV")
        f.addRow("Scale (unit per mV)", self.m_scale)
        f.addRow("Zero", self.m_zero)
        f.addRow(QLabel("value = (mV − zero) × scale"))
        self.tabs.addTab(w, "Manual")

        bb = QDialogButtonBox()
        self.btn_apply = bb.addButton("Apply", QDialogButtonBox.AcceptRole)
        self.btn_remove = bb.addButton("Remove calibration", QDialogButtonBox.DestructiveRole)
        bb.addButton(QDialogButtonBox.Cancel)
        lay.addWidget(bb)
        bb.rejected.connect(self.reject)
        self.btn_apply.clicked.connect(self.apply)
        self.btn_remove.clicked.connect(self.remove)
        self.btn_zero.clicked.connect(lambda: self.capture("zero"))
        self.btn_zero2.clicked.connect(lambda: self.capture("zero"))
        self.btn_load.clicked.connect(lambda: self.capture("load"))
        self.known.valueChanged.connect(self.update_result)
        self.update_result()

    def capture(self, what: str):
        sec = self.avg_s.value()
        for b in (self.btn_zero, self.btn_zero2, self.btn_load):
            b.setEnabled(False)
        lbl = self.lbl_zero if what == "zero" else self.lbl_load
        lbl.setText(f"averaging {sec:g} s…")

        def done(mean, sd, n):
            for b in (self.btn_zero, self.btn_zero2, self.btn_load):
                b.setEnabled(True)
            if n == 0:
                lbl.setText("no data — is the channel enabled and streaming?")
                return
            txt = f"{mean:.6f} mV   (sd {sd * 1e3:.2f} µV, {n} samples)"
            lbl.setText(txt)
            if what == "zero":
                self.zero = mean
                self.lbl_zero2.setText("zero: " + txt)
            else:
                self.loaded = mean
            self.update_result()

        self.main.capture_mv(self.ch, sec, done)

    def update_result(self):
        if self.zero is not None and self.loaded is not None and self.loaded != self.zero:
            sc = self.known.value() / (self.loaded - self.zero)
            self.lbl_res.setText(f"scale = {sc:.6g} {self.unit.currentText()}/mV")
        else:
            self.lbl_res.setText("Capture zero and loaded to compute the scale.")

    def apply(self):
        s = self.main.chan[self.ch]
        tab = self.tabs.currentIndex()
        if tab == 0:
            if self.zero is None or self.loaded is None or self.loaded == self.zero:
                QMessageBox.warning(self, APP_NAME, "Capture both zero and loaded first.")
                return
            s.scale_per_mV = self.known.value() / (self.loaded - self.zero)
            s.zero_mV = self.zero
            s.cal_note = f"known load {self.known.value():g} {self.unit.currentText()}, {time.strftime('%Y-%m-%d %H:%M')}"
        elif tab == 1:
            s.scale_per_mV = self.cap.value() / (self.sens.value() * self.exc.value())
            s.zero_mV = self.zero if self.zero is not None else 0.0
            s.cal_note = f"datasheet {self.cap.value():g} @ {self.sens.value():g} mV/V × {self.exc.value():g} V"
        else:
            if self.m_scale.value() == 0:
                QMessageBox.warning(self, APP_NAME, "Scale cannot be 0.")
                return
            s.scale_per_mV = self.m_scale.value()
            s.zero_mV = self.m_zero.value()
            s.cal_note = "manual"
        s.name = self.name.text().strip() or s.name
        s.cal_unit = self.unit.currentText().strip() or "lbf"
        s.unit = "cal"
        self.accept()

    def remove(self):
        s = self.main.chan[self.ch]
        s.scale_per_mV = 0.0
        s.zero_mV = 0.0
        s.cal_note = ""
        if s.unit == "cal":
            s.unit = "mV"
        self.accept()


# ==========================================================================
# main window
# ==========================================================================
class MainWindow(QMainWindow):
    log_sig = Signal(int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)
        self.worker = DeviceWorker()
        self.dev: Optional[jade.Jade] = None
        self.info: dict = {}
        self.cfg: Optional[jade.Config] = None
        self.gen_cfg: Dict[int, jade.Config] = {}
        self.pending: List = []
        self.pending_since = 0.0
        self.resolving = False
        self.store = SampleStore()
        self.filters: List[jade.LiveFilter] = []
        self.filter_rate = 0.0
        self.t_origin: Optional[float] = None
        self.t_wrap = 0
        self.t_last_raw: Optional[int] = None
        self.last_index: Optional[int] = None
        self.gaps = 0
        self.paused = False
        self.tare = np.zeros(NCH)
        self.markers: List[tuple] = []
        self.recorder: Optional[Recorder] = None
        self.updating = False
        self.meas_rate = 0.0
        # how a port is opened (tests swap in a simulated board)
        self.open_port: Callable[[str], jade.Jade] = lambda port: jade.Jade(port, on_log=self.log_sig.emit)
        self.reconnecting = False
        self.adc_missing = False
        self.desired_cfg: Optional[jade.Config] = None   # settings to put back after a board reboot
        self._lost_at = 0.0
        self._reconnect_tries = 0
        self.t_offset = 0.0
        self._t_last: Optional[float] = None
        self._pending_cfg: Dict[str, tuple] = {}
        self._cfg_lock = threading.Lock()
        self._last_rx = time.monotonic()    # last data or telemetry from the board
        self._last_uptime = None
        self._silent_warned = False
        self._dirty = True
        self._last_draw = 0.0
        self._last_tick_at = time.monotonic()
        self._draw_every = 0.033
        self._rate_n = 0
        self._rate_t = time.monotonic()
        self.sd_files: List[dict] = []
        self.sd_state: dict = {}
        self._sd_busy = False             # a download is running
        self._sd_cancel = False
        self._sd_polling = False
        self._sd_last_poll = 0.0

        self.chan = [ChanSettings(name=f"ch{i}", color=COLORS[i]) for i in range(NCH)]
        self.window_s = 5
        self.layout_mode = "stacked"
        self.lowpass = 0.0
        self.avg = 0
        self.record_dir = str(Path.home() / "Documents" / "Jade")
        self.last_port = ""
        self._geometry = None
        self.load_settings()

        self.log_sig.connect(self._log_device)
        self.build_ui()
        self.rebuild_plots()
        self.refresh_ports()
        self.set_connected(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.slow = QTimer(self)
        self.slow.timeout.connect(self.slow_tick)
        self.slow.start(250)
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self._try_reconnect)

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        # toolbar
        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.port_box = QComboBox()
        self.port_box.setMinimumWidth(260)
        tb.addWidget(QLabel(" Port "))
        tb.addWidget(self.port_box)
        a = tb.addAction("⟳")
        a.setToolTip("Refresh ports")
        a.triggered.connect(self.refresh_ports)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_connect)
        tb.addWidget(self.btn_connect)
        tb.addSeparator()
        self.btn_stream = QPushButton("▶ Stream")
        self.btn_stream.setCheckable(True)
        self.btn_stream.clicked.connect(self.toggle_stream)
        tb.addWidget(self.btn_stream)
        self.btn_pause = QPushButton("Pause view")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setToolTip("Freeze the plot to zoom/pan (data keeps coming in). Space")
        self.btn_pause.clicked.connect(self.set_paused)
        tb.addWidget(self.btn_pause)
        tb.addSeparator()
        self.btn_tare = QPushButton("Tare all")
        self.btn_tare.setToolTip("Zero every shown channel on the last 0.5 s (T)")
        self.btn_tare.clicked.connect(lambda: self.do_tare(None))
        tb.addWidget(self.btn_tare)
        b = QPushButton("Untare")
        b.clicked.connect(self.clear_tare)
        tb.addWidget(b)
        tb.addSeparator()
        self.btn_rec = QPushButton("● Record")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setToolTip("Record every sample to CSV (R)")
        self.btn_rec.clicked.connect(self.toggle_record)
        tb.addWidget(self.btn_rec)
        self.marker_edit = QLineEdit()
        self.marker_edit.setPlaceholderText("marker note, Enter to add (M)")
        self.marker_edit.setMaximumWidth(220)
        self.marker_edit.returnPressed.connect(self.add_marker)
        tb.addWidget(self.marker_edit)
        b = QPushButton("Marker")
        b.clicked.connect(self.add_marker)
        tb.addWidget(b)
        tb.addSeparator()
        tb.addWidget(QLabel(" Window "))
        self.win_box = QComboBox()
        for w in WINDOWS:
            self.win_box.addItem(f"{w} s", w)
        self.win_box.setCurrentIndex(WINDOWS.index(self.window_s) if self.window_s in WINDOWS else 2)
        self.win_box.currentIndexChanged.connect(lambda i: setattr(self, "window_s", WINDOWS[i]))
        tb.addWidget(self.win_box)

        # plots
        self.glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.glw)

        # left: controls
        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_channels_tab(), "Channels")
        self.tabs.addTab(self.build_adc_tab(), "ADC")
        self.tabs.addTab(self.build_display_tab(), "Display")
        self.tabs.addTab(self.build_advanced_tab(), "Advanced")
        self.tabs.addTab(self.build_card_tab(), "Card")
        self.tabs.currentChanged.connect(lambda _: self.sd_refresh(files=True) if self._card_shown() else None)
        self.tabs.setMinimumWidth(600)
        dock = QDockWidget("Controls", self)
        dock.setObjectName("controls")
        dock.setWidget(self.tabs)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # right: readouts
        self.readout_box = QWidget()
        self.readout_lay = QVBoxLayout(self.readout_box)
        self.readout_lay.setContentsMargins(6, 6, 6, 6)
        self.readouts: Dict[int, tuple] = {}
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.readout_box)
        sc.setMinimumWidth(270)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        dock = QDockWidget("Readouts", self)
        dock.setObjectName("readouts")
        dock.setWidget(sc)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.readout_dock = dock

        # bottom: log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFont("Consolas", 9))
        dock = QDockWidget("Log", self)
        dock.setObjectName("log")
        dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.log_dock = dock
        self.resizeDocks([dock], [150], Qt.Vertical)

        # status bar
        sb = self.statusBar()
        self.st_conn = QLabel("disconnected")
        self.st_rate = QLabel("")
        self.st_err = QLabel("")
        self.st_rec = QLabel("")
        for w in (self.st_conn, self.st_rate, self.st_err, self.st_rec):
            w.setContentsMargins(8, 0, 8, 0)
            sb.addPermanentWidget(w)

        # menu
        m = self.menuBar().addMenu("&File")
        m.addAction("Choose recording folder…", self.choose_record_dir)
        m.addAction("Save plot image…", self.save_image)
        m.addSeparator()
        m.addAction("Export profile…", self.export_profile)
        m.addAction("Import profile…", self.import_profile)
        m.addAction("Import cal.json (jade.py)…", self.import_cli_cal)
        m.addSeparator()
        m.addAction("Quit", self.close)
        m = self.menuBar().addMenu("&View")
        for d in self.findChildren(QDockWidget):
            m.addAction(d.toggleViewAction())
        m.addAction("Clear plots", self.clear_data)
        m = self.menuBar().addMenu("&Help")
        m.addAction("Keys", lambda: QMessageBox.information(
            self, APP_NAME, "Space  pause / follow\nT  tare all\nM  add marker\nR  start / stop recording\n\n"
                            "Mouse on a plot: wheel zooms, drag pans, right-click for options.\n"
                            "Pause the view to look back in time; data keeps being recorded."))

        for key, fn in (("Space", lambda: self.btn_pause.click()), ("T", lambda: self.do_tare(None)),
                        ("M", self.add_marker), ("R", lambda: self.btn_rec.click())):
            sc_ = QShortcut(QKeySequence(key), self)
            sc_.setContext(Qt.WindowShortcut)
            sc_.activated.connect(lambda f=fn: None if self._typing() else f())

    def _typing(self) -> bool:
        w = QApplication.focusWidget()
        return isinstance(w, (QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit)) or \
            (isinstance(w, QComboBox) and w.isEditable())

    def build_channels_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        cols = ["Plot", "ADC", "Name", "Gain", "Input", "Unit", "", ""]
        t = QTableWidget(NCH, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setSelectionMode(QTableWidget.NoSelection)
        t.setFocusPolicy(Qt.NoFocus)
        self.ch_table = t
        self.w_show, self.w_en, self.w_name, self.w_gain, self.w_mux, self.w_unit, self.w_val = [], [], [], [], [], [], []
        for ch in range(NCH):
            s = self.chan[ch]
            cb = QCheckBox()
            cb.setChecked(s.show)
            cb.setStyleSheet(f"QCheckBox::indicator:checked {{ background: {s.color}; border: 1px solid #888; }}")
            cb.toggled.connect(lambda v, c=ch: self.on_show(c, v))
            t.setCellWidget(ch, 0, self._center(cb))
            self.w_show.append(cb)
            en = QCheckBox()
            en.toggled.connect(lambda v, c=ch: self.on_enable(c, v))
            t.setCellWidget(ch, 1, self._center(en))
            self.w_en.append(en)
            name = QLineEdit(s.name)
            name.setMinimumWidth(110)
            name.setStyleSheet(f"color: {s.color}; font-weight: bold;")
            name.editingFinished.connect(lambda c=ch, e=name: self.on_name(c, e.text()))
            t.setCellWidget(ch, 2, name)
            self.w_name.append(name)
            g = QComboBox()
            for gv in GAINS:
                g.addItem(f"×{gv}  (±{FS_MV / gv:.4g} mV)", gv)
            g.currentIndexChanged.connect(lambda i, c=ch: self.on_gain(c, GAINS[i]))
            t.setCellWidget(ch, 3, g)
            self.w_gain.append(g)
            mx = QComboBox()
            mx.addItems(MUX_NAMES)
            mx.currentIndexChanged.connect(lambda i, c=ch: self.on_mux(c, i))
            t.setCellWidget(ch, 4, mx)
            self.w_mux.append(mx)
            u = QComboBox()
            u.currentIndexChanged.connect(lambda i, c=ch, box=u: self.on_unit(c, box.itemData(i)))
            t.setCellWidget(ch, 5, u)
            self.w_unit.append(u)
            b = QPushButton("Calibrate…")
            b.clicked.connect(lambda _=False, c=ch: self.open_cal(c))
            t.setCellWidget(ch, 6, b)
            b = QPushButton("Tare")
            b.clicked.connect(lambda _=False, c=ch: self.do_tare(c))
            t.setCellWidget(ch, 7, b)
            self.refresh_unit_box(ch)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        lay.addWidget(t)

        row = QHBoxLayout()
        row.addWidget(QLabel("All gains:"))
        self.all_gain = QComboBox()
        self.all_gain.addItem("—", None)
        for gv in GAINS:
            self.all_gain.addItem(f"×{gv}", gv)
        self.all_gain.activated.connect(self.on_all_gains)
        row.addWidget(self.all_gain)
        b = QPushButton("Plot all")
        b.clicked.connect(lambda: [c.setChecked(True) for c in self.w_show])
        row.addWidget(b)
        b = QPushButton("Plot none")
        b.clicked.connect(lambda: [c.setChecked(False) for c in self.w_show])
        row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        help_ = QLabel("Plot = show on screen.  ADC = channel converting and sent over USB.  "
                       "Gain range is ± full scale at the input.  Input 'Shorted' measures the ADC's own offset/noise.")
        help_.setWordWrap(True)
        help_.setStyleSheet("color: #999;")
        lay.addWidget(help_)
        return w

    @staticmethod
    def _center(widget: QWidget) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.addWidget(widget)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(0, 0, 0, 0)
        return box

    def build_adc_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Conversion")
        f = QFormLayout(g)
        self.osr_box = QComboBox()
        for o in OSRS:
            self.osr_box.addItem(f"{o}   ({4.096e6 / o:,.0f} SPS)", o)
        self.osr_box.currentIndexChanged.connect(lambda i: self.on_osr(OSRS[i]))
        f.addRow("OSR / data rate", self.osr_box)
        f.addRow("Power mode", QLabel("High resolution (only mode valid with 8.192 MHz CLKIN)"))
        self.gchop = QCheckBox("Global chop (lower offset drift, lower rate)")
        self.gchop.toggled.connect(lambda _: self.on_gchop())
        f.addRow(self.gchop)
        self.gchop_dly = QSpinBox()
        self.gchop_dly.setRange(0, 15)
        self.gchop_dly.valueChanged.connect(lambda _: self.on_gchop())
        f.addRow("Chop delay code", self.gchop_dly)
        self.dcblock = QComboBox()
        self.dcblock.addItem("Off", 0)
        for c in range(1, 16):
            self.dcblock.addItem(f"code {c}", c)
        self.dcblock.currentIndexChanged.connect(lambda i: self.on_dcblock())
        f.addRow("DC block (high-pass)", self.dcblock)
        self.autostart = QCheckBox("Start streaming as soon as a host opens the port")
        self.autostart.toggled.connect(self.on_autostart)
        f.addRow(self.autostart)
        lay.addWidget(g)

        g = QGroupBox("Device")
        grid = QGridLayout(g)
        buttons = [
            ("Save config to flash", self.dev_save), ("Load from flash", self.dev_load),
            ("Factory defaults", self.dev_defaults), ("Sync (restart filters)", self.dev_sync),
            ("Reset ADC", self.dev_reset), ("Identify (blink LEDs)", self.dev_identify),
            ("Reboot", lambda: self.dev_reboot(False)), ("Reboot to USB bootloader", lambda: self.dev_reboot(True)),
        ]
        self.dev_buttons = {}
        for i, (txt, fn) in enumerate(buttons):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
            self.dev_buttons[txt] = b
        lay.addWidget(g)

        g = QGroupBox("Board")
        f = QFormLayout(g)
        self.info_labels = {}
        for k in ("firmware", "serial", "adc_id", "spi_hz", "flash"):
            lbl = QLabel("—")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            f.addRow(k, lbl)
            self.info_labels[k] = lbl
        lay.addWidget(g)
        lay.addStretch()
        return w

    def build_display_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Layout")
        v = QVBoxLayout(g)
        self.rb_stacked = QRadioButton("One plot per channel")
        self.rb_overlay = QRadioButton("All channels on one plot")
        (self.rb_stacked if self.layout_mode == "stacked" else self.rb_overlay).setChecked(True)
        self.rb_stacked.toggled.connect(self.on_layout)
        v.addWidget(self.rb_stacked)
        v.addWidget(self.rb_overlay)
        lay.addWidget(g)

        g = QGroupBox("Filter (display and readouts; recordings stay raw)")
        f = QFormLayout(g)
        self.lp_spin = QDoubleSpinBox()
        self.lp_spin.setRange(0, 20000)
        self.lp_spin.setDecimals(1)
        self.lp_spin.setSuffix(" Hz")
        self.lp_spin.setSpecialValueText("off")
        self.lp_spin.setValue(self.lowpass)
        self.lp_spin.setEnabled(HAVE_SCIPY)
        if not HAVE_SCIPY:
            self.lp_spin.setToolTip("pip install scipy to enable")
        self.lp_spin.editingFinished.connect(self.on_filter)
        f.addRow("Low-pass (4th order)", self.lp_spin)
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(0, 100000)
        self.avg_spin.setSpecialValueText("off")
        self.avg_spin.setValue(self.avg)
        self.avg_spin.setSuffix(" samples")
        self.avg_spin.editingFinished.connect(self.on_filter)
        f.addRow("Moving average", self.avg_spin)
        presets = QHBoxLayout()
        for label, lp, av in (("None", 0, 0), ("5 Hz", 5, 0), ("20 Hz", 20, 0), ("100 Hz", 100, 0)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, lp=lp, av=av: self.set_filter(lp, av))
            b.setEnabled(HAVE_SCIPY or lp == 0)
            presets.addWidget(b)
        f.addRow("Presets", presets)
        self.lbl_filter = QLabel("")
        self.lbl_filter.setWordWrap(True)
        self.lbl_filter.setStyleSheet("color: #999;")
        f.addRow(self.lbl_filter)
        lay.addWidget(g)

        g = QGroupBox("Stats")
        v = QVBoxLayout(g)
        self.stats_on_window = QRadioButton("Mean / p-p / σ over the visible window")
        self.stats_on_1s = QRadioButton("… over the last second")
        self.stats_on_window.setChecked(True)
        v.addWidget(self.stats_on_window)
        v.addWidget(self.stats_on_1s)
        lay.addWidget(g)

        b = QPushButton("Clear plots")
        b.clicked.connect(self.clear_data)
        lay.addWidget(b)
        lay.addStretch()
        self.update_filter_label()
        return w

    def build_advanced_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Per-channel ADC trim (applied inside the ADC)")
        v = QVBoxLayout(g)
        t = QTableWidget(NCH, 4)
        t.setHorizontalHeaderLabels(["Phase (mod. clk)", "Offset cal (codes)", "Gain cal (×)", "DC block"])
        t.verticalHeader().setDefaultSectionSize(26)
        self.adv_phase, self.adv_ocal, self.adv_gcal, self.adv_dcdis = [], [], [], []
        for ch in range(NCH):
            p = QSpinBox()
            p.setRange(-512, 511)
            t.setCellWidget(ch, 0, p)
            o = QSpinBox()
            o.setRange(-8388608, 8388607)
            t.setCellWidget(ch, 1, o)
            gc = QDoubleSpinBox()
            gc.setRange(0, 1.99999988)
            gc.setDecimals(7)
            gc.setSingleStep(0.0001)
            t.setCellWidget(ch, 2, gc)
            d = QCheckBox("enabled")
            t.setCellWidget(ch, 3, d)
            self.adv_phase.append(p)
            self.adv_ocal.append(o)
            self.adv_gcal.append(gc)
            self.adv_dcdis.append(d)
        t.setVerticalHeaderLabels([f"ch{c}" for c in range(NCH)])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(t)
        b = QPushButton("Apply trim to ADC")
        b.clicked.connect(self.on_apply_trim)
        v.addWidget(b)
        lay.addWidget(g)

        g = QGroupBox("Registers (ADS131M08)")
        f = QGridLayout(g)
        self.reg_addr = QSpinBox()
        self.reg_addr.setRange(0, 0x3F)
        self.reg_addr.setDisplayIntegerBase(16)
        self.reg_addr.setPrefix("0x")
        self.reg_val = QLineEdit()
        self.reg_val.setPlaceholderText("hex value, e.g. FF0E")
        f.addWidget(QLabel("Address"), 0, 0)
        f.addWidget(self.reg_addr, 0, 1)
        f.addWidget(QLabel("Value"), 0, 2)
        f.addWidget(self.reg_val, 0, 3)
        b = QPushButton("Read")
        b.clicked.connect(self.on_reg_read)
        f.addWidget(b, 1, 0)
        b = QPushButton("Write")
        b.clicked.connect(self.on_reg_write)
        f.addWidget(b, 1, 1)
        b = QPushButton("Dump all to log")
        b.clicked.connect(self.on_reg_dump)
        f.addWidget(b, 1, 2, 1, 2)
        lay.addWidget(g)
        lay.addStretch()
        return w


    def build_card_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("SD card")
        f = QFormLayout(g)
        self.sd_labels = {}
        for k, label in (("state", "State"), ("card", "Card"), ("file", "Logging to"),
                         ("health", "Problems"), ("message", "Board says")):
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            f.addRow(label, lbl)
            self.sd_labels[k] = lbl
        self.sd_log_cb = QCheckBox("Log to the card (from power-up, no PC needed)")
        self.sd_log_cb.setToolTip("On by default. \"Save config to flash\" (ADC tab) keeps this setting "
                                  "after a power cycle.")
        self.sd_log_cb.toggled.connect(self.on_sd_logging)
        f.addRow(self.sd_log_cb)
        lay.addWidget(g)

        g = QGroupBox("Files on the card")
        v = QVBoxLayout(g)
        t = QTableWidget(0, 3)
        t.setHorizontalHeaderLabels(["Name", "Size", ""])
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(22)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        t.setMinimumHeight(180)
        t.itemSelectionChanged.connect(self.sd_update_buttons)
        t.doubleClicked.connect(lambda _: self.sd_download())
        self.sd_table = t
        v.addWidget(t)
        row = QHBoxLayout()
        self.sd_btn_refresh = QPushButton("Refresh")
        self.sd_btn_refresh.clicked.connect(lambda: self.sd_refresh(files=True))
        self.sd_btn_get = QPushButton("Download…")
        self.sd_btn_get.setToolTip("Copy the selected files to the PC (double-click a file works too)")
        self.sd_btn_get.clicked.connect(self.sd_download)
        self.sd_btn_del = QPushButton("Delete")
        self.sd_btn_del.clicked.connect(self.sd_delete)
        self.sd_btn_clear = QPushButton("Delete all logs…")
        self.sd_btn_clear.setToolTip("Every JADEnnnn.BIN (and OPALnnnn.BIN from firmware 2.0). Other files stay. "
                                     "Logging carries on in a new JADE0001.BIN.")
        self.sd_btn_clear.clicked.connect(self.sd_clear)
        for b in (self.sd_btn_refresh, self.sd_btn_get, self.sd_btn_del, self.sd_btn_clear):
            row.addWidget(b)
        v.addLayout(row)
        self.sd_csv = QCheckBox("Also convert logs to CSV, with the calibration set in this app")
        self.sd_csv.setChecked(True)
        v.addWidget(self.sd_csv)
        row = QHBoxLayout()
        self.sd_progress = QProgressBar()
        self.sd_progress.setTextVisible(True)
        self.sd_btn_cancel = QPushButton("Cancel")
        self.sd_btn_cancel.clicked.connect(lambda: setattr(self, "_sd_cancel", True))
        row.addWidget(self.sd_progress)
        row.addWidget(self.sd_btn_cancel)
        v.addLayout(row)
        self.sd_progress.hide()
        self.sd_btn_cancel.hide()
        lay.addWidget(g)

        g = QGroupBox("USB drive")
        v = QVBoxLayout(g)
        self.sd_btn_drive = QPushButton("Show the card as a USB drive")
        self.sd_btn_drive.setCheckable(True)
        self.sd_btn_drive.clicked.connect(self.sd_toggle_drive)
        v.addWidget(self.sd_btn_drive)
        lbl = QLabel("The card shows up on the PC like a USB stick, for big copies, other files or "
                     "formatting. Logging pauses while the PC has it. Eject the drive on the PC when "
                     "you're done: the board takes the card back and starts a new log file. "
                     "Streaming to this app keeps going, with short pauses while the PC reads or writes.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #999;")
        v.addWidget(lbl)
        lay.addWidget(g)
        lay.addStretch()
        self.sd_update_buttons()
        return w

    # ------------------------------------------------------------ SD card
    def _card_shown(self) -> bool:
        return self.tabs.currentIndex() == self.tabs.count() - 1

    @staticmethod
    def _size(n: int) -> str:
        for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
            if n >= div:
                return f"{n / div:.1f} {unit}"
        return f"{n} B"

    def sd_refresh(self, files: bool = False):
        """Status (and the file list) from the board, in the background."""
        if not self.dev or self._sd_polling:
            return
        dev = self.dev
        self._sd_polling = True
        self._sd_last_poll = time.monotonic()

        def job():
            st = dev.sd_status()
            lst = None
            if (files or st["state"] == 1) and st["state"] in (1, 2):
                lst = dev.sd_list()
            elif st["state"] not in (1, 2):
                lst = []
            return st, lst

        def ok(r):
            self._sd_polling = False
            self.sd_show(*r)

        def err(e):
            self._sd_polling = False
            if files:
                self.fail("SD card")(e)

        self.worker.run(job, ok, err)

    def sd_show(self, st: dict, files: Optional[List[dict]]):
        self.sd_state = st
        L = self.sd_labels
        colors = {0: "#999", 1: "#81c784", 2: "#ddd", 3: "#4fc3f7", 4: "#ef5350"}
        L["state"].setText(st["state_text"])
        L["state"].setStyleSheet(f"color: {colors.get(st['state'], '#ddd')};")
        if st["card_bytes"]:
            txt = f"{st['card']} {self._size(st['card_bytes'])}"
            if st["fs"]:
                txt += f", {st['fs']}, {self._size(st['free_bytes'])} free"
            txt += f"  ({st['bus_width']}-bit, {st['clock_hz'] / 1e6:.0f} MHz)"
            L["card"].setText(txt)
        else:
            L["card"].setText("—")
        L["file"].setText(f"{st['file']}  ({self._size(st['bytes_written'])})" if st["file"] else "—")
        L["health"].setText(f"{st['dropped_samples']} samples (card too slow), {st['write_errors']} write error(s)"
                            if st["dropped_samples"] or st["write_errors"] else "none")
        L["message"].setText(st["message"])
        self.sd_log_cb.blockSignals(True)
        self.sd_log_cb.setChecked(st["logging_on"])
        self.sd_log_cb.blockSignals(False)
        self.sd_btn_drive.setChecked(st["state"] == 3)
        self.sd_btn_drive.setText("Take the card back from the PC" if st["state"] == 3
                                  else "Show the card as a USB drive")
        if files is not None:
            keep = {self.sd_table.item(r.row(), 0).text() for r in self.sd_table.selectionModel().selectedRows()}
            files = sorted(files, key=lambda x: (not x["log"], x["name"]))
            self.sd_files = files
            t = self.sd_table
            t.clearSelection()
            t.setRowCount(len(files))
            for i, fi in enumerate(files):
                t.setItem(i, 0, QTableWidgetItem(fi["name"]))
                it = QTableWidgetItem(self._size(fi["size"]))
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, 1, it)
                t.setItem(i, 2, QTableWidgetItem("being written" if fi["active"] else ""))
                if fi["name"] in keep:
                    t.selectRow(i)
        self.sd_update_buttons()

    def sd_selected(self) -> List[dict]:
        rows = sorted({r.row() for r in self.sd_table.selectionModel().selectedRows()})
        return [self.sd_files[r] for r in rows if r < len(self.sd_files)]

    def sd_update_buttons(self):
        st = self.sd_state.get("state")
        card_ok = bool(self.dev) and st in (1, 2) and not self._sd_busy
        sel = self.sd_selected() if hasattr(self, "sd_table") else []
        self.sd_btn_get.setEnabled(card_ok and bool(sel))
        self.sd_btn_del.setEnabled(card_ok and bool(sel) and not any(f["active"] for f in sel))
        self.sd_btn_del.setToolTip("The file being written can't be deleted: turn logging off first"
                                   if any(f["active"] for f in sel) else "")
        self.sd_btn_clear.setEnabled(card_ok and any(f["log"] for f in self.sd_files))
        self.sd_btn_refresh.setEnabled(bool(self.dev) and not self._sd_busy)
        self.sd_btn_drive.setEnabled(bool(self.dev) and not self._sd_busy and st in (1, 2, 3))
        self.sd_log_cb.setEnabled(bool(self.dev) and st != 3)

    def on_sd_logging(self, v):
        if self.dev and self.cfg and not self.updating:
            self.cfg_call("SD logging " + ("on" if v else "off"), self.dev.sd_logging, v, key="sdlog")
            QTimer.singleShot(600, lambda: self.sd_refresh(files=True))

    def sd_download(self):
        sel = self.sd_selected()
        if not sel or not self.dev or self._sd_busy:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save to folder", self.record_dir)
        if not folder:
            return
        convert = self.sd_csv.isChecked()
        cal = {ch: (c.zero_mV, c.scale_per_mV, c.cal_unit or "cal")
               for ch, c in enumerate(self.chan) if c.calibrated}
        dev = self.dev
        total = sum(f["size"] for f in sel)
        self._sd_busy = True
        self._sd_cancel = False
        self.sd_progress.setRange(0, 1000)
        self.sd_progress.setValue(0)
        self.sd_progress.show()
        self.sd_btn_cancel.show()
        self.sd_update_buttons()
        post = self.worker.bridge.call.emit

        def show(done, name):
            self.sd_progress.setValue(int(1000 * done / max(1, total)))
            self.sd_progress.setFormat(f"{name}  {self._size(done)} / {self._size(total)}")

        def run():
            msgs, base, t0 = [], 0, time.monotonic()
            try:
                for fi in sel:
                    path = Path(folder) / fi["name"]

                    def prog(d, _s, fi=fi, base=base):
                        post(lambda: show(base + d, fi["name"]))
                        return not self._sd_cancel
                    n = dev.sd_download(fi["name"], path, size=fi["size"], progress=prog)
                    base += fi["size"]
                    if self._sd_cancel:
                        msgs.append((2, f"download cancelled: {path} is incomplete ({n} of {fi['size']} bytes)"))
                        break
                    msgs.append((1, f"downloaded {fi['name']} to {path}"))
                    if convert and fi["log"]:
                        post(lambda fi=fi: self.sd_progress.setFormat(f"{fi['name']}: converting to CSV…"))
                        import jade_sd2csv
                        summary, _, _ = jade_sd2csv.convert(path, path.with_suffix(".csv"), cal)
                        msgs.append((1, summary + f" -> {path.with_suffix('.csv').name}"))
                dt = max(1e-3, time.monotonic() - t0)
                msgs.append((1, f"{self._size(base)} in {dt:.1f} s ({self._size(int(base / dt))}/s)"))
            except Exception as e:  # noqa: BLE001
                msgs.append((3, f"download failed: {e}"))
            post(lambda: finish(msgs))

        def finish(msgs):
            for lvl, m in msgs:
                self.log(m, lvl)
            self._sd_busy = False
            self.sd_progress.hide()
            self.sd_btn_cancel.hide()
            self.sd_update_buttons()

        threading.Thread(target=run, daemon=True).start()

    def sd_delete(self):
        sel = [f for f in self.sd_selected() if not f["active"]]
        if not sel or not self.dev:
            return
        names = ", ".join(f["name"] for f in sel[:5]) + (f" and {len(sel) - 5} more" if len(sel) > 5 else "")
        if QMessageBox.question(self, APP_NAME, f"Delete {names} from the card?") != QMessageBox.Yes:
            return
        dev = self.dev

        def job():
            for f in sel:
                dev.sd_delete(f["name"])
            return len(sel)
        self.worker.run(job, lambda n: (self.log(f"deleted {n} file(s) from the card"), self.sd_refresh(files=True)),
                        lambda e: (self.fail("delete")(e), self.sd_refresh(files=True)))

    def sd_clear(self):
        logs = [f for f in self.sd_files if f["log"]]
        if not self.dev or not logs:
            return
        size = sum(f["size"] for f in logs)
        if QMessageBox.question(self, APP_NAME, f"Delete every log file on the card ({len(logs)} files, "
                                f"{self._size(size)})?\n\nOther files stay. If the board is logging, it "
                                "carries on in a new JADE0001.BIN.") != QMessageBox.Yes:
            return
        dev = self.dev
        self.worker.run(dev.sd_delete_all_logs,
                        lambda n: (self.log(f"deleted {n} log file(s) from the card"),
                                   QTimer.singleShot(300, lambda: self.sd_refresh(files=True))),
                        self.fail("delete all logs"))

    def sd_toggle_drive(self):
        if not self.dev:
            return
        on = self.sd_btn_drive.isChecked()
        if not on and QMessageBox.question(
                self, APP_NAME, "Eject the drive on the PC first, so nothing is left half-written.\n\n"
                                "Take the card back now anyway?") != QMessageBox.Yes:
            self.sd_btn_drive.setChecked(True)
            return
        dev = self.dev

        def ok(_):
            self.log("the card is now a USB drive on this PC; logging paused until you eject it" if on
                     else "the board has the card back")
            self.sd_refresh(files=True)
        self.worker.run(lambda: dev.sd_drive(on), ok,
                        lambda e: (self.fail("USB drive")(e), self.sd_refresh(files=True)))

    # ------------------------------------------------------------- plots
    def rebuild_plots(self):
        self.glw.clear()
        self.plots: Dict[int, pg.PlotItem] = {}
        self.curves: Dict[int, pg.PlotDataItem] = {}
        shown = [ch for ch in range(NCH) if self.chan[ch].show]
        self.layout_mode = "stacked" if self.rb_stacked.isChecked() else "overlay"
        if not shown:
            self.glw.addLabel("No channel selected for plotting (Channels tab → Plot)")
            self.rebuild_readouts()
            return
        if self.layout_mode == "stacked":
            first = None
            for i, ch in enumerate(shown):
                p = self.glw.addPlot(row=i, col=0)
                self._setup_plot(p)
                p.setLabel("left", self.chan[ch].name, units=None, color=self.chan[ch].color)
                if first is None:
                    first = p
                else:
                    p.setXLink(first)
                if i < len(shown) - 1:
                    p.getAxis("bottom").setStyle(showValues=False)
                else:
                    p.setLabel("bottom", "time", units="s")
                self.plots[ch] = p
                self.curves[ch] = self._curve(p, ch)
        else:
            p = self.glw.addPlot(row=0, col=0)
            self._setup_plot(p)
            p.addLegend(offset=(10, 10))
            p.setLabel("bottom", "time", units="s")
            for ch in shown:
                self.plots[ch] = p
                self.curves[ch] = self._curve(p, ch, name=self.chan[ch].name)
        self.marker_lines = []
        for t, text in self.markers:
            self._draw_marker(t, text)
        self.update_axis_labels()
        self.rebuild_readouts()

    def _setup_plot(self, p: pg.PlotItem):
        p.getAxis("left").setWidth(78)
        p.getAxis("left").enableAutoSIPrefix(False)
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setClipToView(True)
        p.setDownsampling(auto=True, mode="peak")
        p.enableAutoRange(axis="y")
        p.setMouseEnabled(x=True, y=True)
        p.getViewBox().sigRangeChangedManually.connect(lambda *_: self.set_paused(True))

    def _curve(self, p: pg.PlotItem, ch: int, name=None) -> pg.PlotDataItem:
        c = p.plot(pen=pg.mkPen(self.chan[ch].color, width=1), name=name)
        c.setDownsampling(auto=True, method="peak")
        c.setClipToView(True)
        return c

    def update_axis_labels(self):
        if self.layout_mode == "stacked":
            for ch, p in self.plots.items():
                p.setLabel("left", f"{self.chan[ch].name} [{self.chan[ch].unit_label()}]", color=self.chan[ch].color)
        elif self.plots:
            units = {self.chan[ch].unit_label() for ch in self.plots}
            next(iter(self.plots.values())).setLabel("left", " / ".join(sorted(units)))

    def rebuild_readouts(self):
        while self.readout_lay.count():
            it = self.readout_lay.takeAt(0)
            wdg = it.widget()
            if wdg:
                wdg.hide()
                wdg.setParent(None)
                wdg.deleteLater()
        self.readouts = {}
        for ch in range(NCH):
            if not self.chan[ch].show:
                continue
            fr = QFrame()
            fr.setFrameShape(QFrame.StyledPanel)
            fr.setStyleSheet(f"QFrame {{ border-left: 4px solid {self.chan[ch].color}; }}")
            v = QVBoxLayout(fr)
            v.setSpacing(0)
            name = QLabel()
            name.setStyleSheet(f"color: {self.chan[ch].color}; border: none; font-weight: bold;")
            val = QLabel("—")
            val.setStyleSheet("border: none;")
            val.setFont(QFont("Consolas", 18, QFont.Bold))
            val.setAlignment(Qt.AlignRight)
            stats = QLabel("")
            stats.setStyleSheet("border: none; color: #aaa;")
            stats.setFont(QFont("Consolas", 8))
            stats.setAlignment(Qt.AlignRight)
            stats.setWordWrap(True)
            v.addWidget(name)
            v.addWidget(val)
            v.addWidget(stats)
            self.readout_lay.addWidget(fr)
            self.readouts[ch] = (name, val, stats)
        self.readout_lay.addStretch()
        self.refresh_readout_names()

    def refresh_readout_names(self):
        for ch, (name, _, _) in self.readouts.items():
            s = self.chan[ch]
            name.setText(f"{s.name}   [{s.unit_label()}]" + ("   tared" if self.tare[ch] else ""))

    # ------------------------------------------------------------ logging
    def log(self, msg: str, level: int = 1):
        tag = ["DBG", "INF", "WRN", "ERR"][level & 3]
        color = ["#888", "#ddd", "#ffb74d", "#ef5350"][level & 3]
        self.log_view.appendHtml(f'<span style="color:{color}">{time.strftime("%H:%M:%S")} {tag} '
                                 f'{msg.replace("<", "&lt;")}</span>')
        if level >= 2:
            self.statusBar().showMessage(msg, 6000)

    @Slot(int, str)
    def _log_device(self, level: int, msg: str):
        self.log("board: " + msg, level)

    def fail(self, what: str):
        def cb(e: Exception):
            # no modal dialog: the log + status bar say it, the stream keeps running
            self.log(f"{what} refused: {e}" if isinstance(e, jade.JadeError) else f"{what}: {e}", 3)
            if self.cfg:
                self.apply_cfg_to_widgets(self.cfg)
        return cb

    # --------------------------------------------------------- connection
    def refresh_ports(self):
        self.port_box.clear()
        if list_ports is None:
            return
        ports = list(list_ports.comports())
        best = jade.find_port()
        for p in sorted(ports, key=lambda p: (not (p.vid == jade.USB_VID and p.pid == jade.USB_PID), p.device)):
            tag = ""
            if p.vid == jade.USB_VID and p.pid == jade.USB_PID:
                tag = "Jade Data" if p.device == best else "Jade (console?)"
            self.port_box.addItem(f"{p.device}  {tag or p.description}", p.device)
        for i in range(self.port_box.count()):
            if self.port_box.itemData(i) in (best, self.last_port):
                self.port_box.setCurrentIndex(i)
                if self.port_box.itemData(i) == best:
                    break

    def toggle_connect(self):
        if self.reconnecting:
            self.stop_reconnecting()
            return
        if self.dev:
            self.disconnect_device()
            return
        port = self.port_box.currentData()
        if not port:
            QMessageBox.warning(self, APP_NAME, "No port selected. Plug the board in and press ⟳.")
            return
        self.last_port = port
        self.btn_connect.setEnabled(False)
        self.log(f"connecting to {port}…")
        self.worker.run(lambda: self.open_port(port), self.attach_device, self._connect_failed)

    def _connect_failed(self, e):
        self.btn_connect.setEnabled(True)
        self.log(f"cannot open port: {e}", 3)
        QMessageBox.warning(self, APP_NAME, f"Cannot open the port:\n{e}\n\n"
                            "Close any other program using it (SerialPlot, a terminal…) and check the board "
                            "is running (green LED D7 blinking).")

    def attach_device(self, dev: jade.Jade, resume: bool = False):
        """resume=True after an automatic reconnect: keep the plots, the
        recording and the user's settings."""
        self.dev = dev

        def setup():
            info = dev.ping()
            cfg = dev.get_config()
            gen = dev.get_telemetry()["gen"]
            streaming = False
            if info["adc_ok"]:            # without the ADC the board refuses to stream
                dev.stream_start()
                streaming = True
            return info, cfg, gen, streaming

        def ok(res):
            info, cfg, gen, streaming = res
            self.adc_missing = not info["adc_ok"]
            self.info = info
            for k, lbl in self.info_labels.items():
                lbl.setText(str(info.get(k, "—")))
            self.dev_buttons["Save config to flash"].setEnabled(info.get("flash", False))
            self.dev_buttons["Load from flash"].setEnabled(info.get("flash", False))
            self.gen_cfg.clear()
            self.pending = []
            if resume:
                # board time restarted: continue the time axis from where it stopped
                st = self.store
                last_t = float(st.t[st.n - 1]) if st.n else 0.0
                self.t_offset = last_t + max(0.001, time.monotonic() - self._lost_at)
                self.t_origin = None
                self.t_wrap = 0
                self.t_last_raw = None
                self.last_index = None
                self.rebuild_filters(refilter=False)
                if self.recorder:
                    self.recorder.comment(f"board reconnected after {time.monotonic() - self._lost_at:.1f} s "
                                          "(samples missing in between)")
                self.add_auto_marker("reconnected")
            else:
                self.clear_data()
            want = self.desired_cfg if resume else None
            self._last_rx = time.monotonic()
            self._silent_warned = False
            self.reconnecting = False
            self.reconnect_timer.stop()
            self.on_config((cfg, gen))
            self.set_connected(True)
            self.btn_stream.setChecked(streaming)
            self.btn_stream.setText("■ Stop" if streaming else "▶ Stream")
            if info["adc_ok"]:
                self.log(f"connected: firmware {info['firmware']}, ADC {info['adc_id']}", 1)
            else:
                self.log(f"connected: firmware {info['firmware']}, but the ADS131M08 is NOT DETECTED "
                         f"(ID read {info['adc_id']}, expected 0x28xx). {adc_id_hint(info['adc_id'])} "
                         "The board keeps retrying; streaming starts by itself once the ADC answers.", 3)
            # a rebooted board is back on its defaults: put the user's settings back
            if resume and want is not None and want.pack() != cfg.pack():
                self.cfg_call("settings restored after the board restarted", dev.set_config, want, key="restore")

        def err(e):
            self.log(f"board not answering: {e}", 3)
            if resume:
                self._lose_device(f"no answer after reconnecting ({e})")
            else:
                self.disconnect_device()

        self.worker.run(setup, ok, err)

    def disconnect_device(self):
        """User-requested disconnect (no automatic reconnect)."""
        self.stop_reconnecting(quiet=True)
        if self.recorder:
            self.btn_rec.setChecked(False)
            self.toggle_record()
        dev, self.dev = self.dev, None
        self.set_connected(False)
        if dev:
            def close():
                try:
                    dev.command(jade.CMD["STREAM_STOP"], timeout=0.5)
                except Exception:  # noqa: BLE001
                    pass
                dev.close()
            self.worker.run(close, lambda _: self.log("disconnected"))

    def _lose_device(self, why: str):
        """Connection dropped (board rebooted, unplugged, USB glitch): keep
        everything and try to get it back every second."""
        dev, self.dev = self.dev, None
        if self.cfg is not None:
            self.desired_cfg = self.cfg
        self._lost_at = time.monotonic()
        if dev:
            self.worker.run(lambda: dev.close())
        if self.recorder:
            self.recorder.comment(f"connection lost: {why}")
        self.reconnecting = True
        self._reconnect_tries = 0
        self.set_connected(False)
        self.btn_connect.setText("Stop reconnecting")
        self.st_conn.setText("reconnecting…")
        self.st_conn.setStyleSheet("color: #ffb74d; font-weight: bold;")
        self.log(f"connection lost: {why}. Reconnecting automatically "
                 "(the board reboots by itself after a crash).", 2)
        crash_log(f"connection lost: {why}")
        self.reconnect_timer.start(1000)

    def _try_reconnect(self):
        if not self.reconnecting or self.dev is not None or self.worker.busy:
            return
        port = self.last_port
        self._reconnect_tries += 1
        if self._reconnect_tries % 15 == 0:
            self.log(f"still trying to reconnect to {port}… (board stuck? unplug/replug, or reload the .uf2)", 2)

        def ok(dev):
            if self.reconnecting and self.dev is None:
                self.attach_device(dev, resume=True)
            else:
                dev.close()

        self.worker.run(lambda: self.open_port(port), ok, lambda e: None)

    def stop_reconnecting(self, quiet: bool = False):
        if not self.reconnecting:
            return
        self.reconnecting = False
        self.reconnect_timer.stop()
        if self.recorder:
            self.btn_rec.setChecked(False)
            self.toggle_record()
        self.set_connected(False)
        if not quiet:
            self.log("stopped reconnecting")

    def set_connected(self, on: bool):
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText("Disconnect" if on else ("Stop reconnecting" if self.reconnecting else "Connect"))
        self.port_box.setEnabled(not on and not self.reconnecting)
        for w in (self.btn_stream, self.tabs.widget(0), self.tabs.widget(1), self.tabs.widget(3),
                  self.tabs.widget(4)):
            w.setEnabled(on)
        if not on:
            self.sd_state = {}
            self.sd_files = []
            self.sd_table.setRowCount(0)
            for lbl in self.sd_labels.values():
                lbl.setText("—")
                lbl.setStyleSheet("")
        else:
            self.sd_refresh(files=True)
        self.sd_update_buttons()
        self.btn_rec.setEnabled(on or self.recorder is not None)
        if not on:
            self.btn_stream.setChecked(False)
            self.st_conn.setText("disconnected")

    def toggle_stream(self):
        if not self.dev:
            return
        on = self.btn_stream.isChecked()
        self.btn_stream.setText("■ Stop" if on else "▶ Stream")

        def failed(e):
            self.btn_stream.setChecked(False)
            self.btn_stream.setText("▶ Stream")
            self.fail("stream")(e)
        self.worker.run((self.dev.stream_start if on else self.dev.stream_stop),
                        lambda _: self.log("streaming" if on else "stream stopped"), failed)

    # ------------------------------------------------------ config round trip
    def cfg_call(self, what: str, fn: Callable, *args, key: Optional[str] = None):
        """Run a config-changing command in the background and learn the new
        config generation so samples are converted with the right gain.

        Calls with the same `key` are coalesced: if one is still waiting its
        turn, only the latest values are sent (fast clicking or scrolling
        through a combo box no longer piles up commands)."""
        if not self.dev:
            return
        dev = self.dev
        if key is not None:
            with self._cfg_lock:
                if key in self._pending_cfg:
                    self._pending_cfg[key] = (what, fn, args)
                    return
                self._pending_cfg[key] = (what, fn, args)

        def job():
            if key is not None:
                with self._cfg_lock:
                    w, f, a = self._pending_cfg.pop(key)
            else:
                w, f, a = what, fn, args
            cfg = f(*a)
            return w, (cfg, dev.get_telemetry()["gen"])

        def ok(r):
            w, res = r
            self.on_config(res)
            self.log(w)

        self.worker.run(job, ok, self.fail(what))

    def on_config(self, res):
        cfg, gen = res
        self.cfg = cfg
        self.desired_cfg = cfg
        self.gen_cfg[gen] = cfg
        if len(self.gen_cfg) > 64:
            for k in list(self.gen_cfg)[:-32]:
                del self.gen_cfg[k]
        with self._cfg_lock:
            more_coming = bool(self._pending_cfg)
        if not more_coming:              # don't flicker the widgets mid-burst
            self.apply_cfg_to_widgets(cfg)

    def apply_cfg_to_widgets(self, cfg: jade.Config):
        self.updating = True
        try:
            for ch in range(NCH):
                self.w_en[ch].setChecked(bool(cfg.ch_enable >> ch & 1))
                self.w_gain[ch].setCurrentIndex(GAINS.index(cfg.gain[ch]))
                self.w_mux[ch].setCurrentIndex(cfg.mux[ch])
                self.adv_phase[ch].setValue(cfg.phase[ch])
                self.adv_ocal[ch].setValue(cfg.ocal[ch])
                self.adv_gcal[ch].setValue(cfg.gcal[ch] / 0x800000)
                self.adv_dcdis[ch].setChecked(not (cfg.dc_block_dis >> ch & 1))
            self.osr_box.setCurrentIndex(OSRS.index(cfg.osr))
            self.gchop.setChecked(bool(cfg.gchop_en))
            self.gchop_dly.setValue(cfg.gchop_delay)
            self.dcblock.setCurrentIndex(cfg.dc_block)
            self.autostart.setChecked(bool(cfg.autostart))
            self.sd_log_cb.setChecked(not cfg.sd_log_off)
        finally:
            self.updating = False

    def on_enable(self, ch, v):
        if self.updating or not self.cfg:
            return
        mask = sum(1 << c for c in range(NCH) if self.w_en[c].isChecked())
        self.cfg_call(f"channels 0x{mask:02X}", self.dev.set_channels, mask, key="mask")

    def on_gain(self, ch, g):
        if not self.updating and self.cfg:
            self.cfg_call(f"ch{ch} gain {g}", self.dev.set_gain, ch, g, key=f"gain{ch}")

    def on_all_gains(self, i):
        g = self.all_gain.itemData(i)
        if g and self.cfg:
            self.cfg_call(f"all gains {g}", self.dev.set_gain, jade.CH_ALL, g, key="gain_all")
        self.all_gain.setCurrentIndex(0)

    def on_mux(self, ch, m):
        if not self.updating and self.cfg:
            self.cfg_call(f"ch{ch} input {MUX_NAMES[m]}", self.dev.set_mux, ch, m, key=f"mux{ch}")

    def on_osr(self, osr):
        if not self.updating and self.cfg:
            self.cfg_call(f"OSR {osr}", self.dev.set_osr, osr, key="osr")

    def on_gchop(self):
        if not self.updating and self.cfg:
            self.cfg_call("global chop", self.dev.set_gchop, self.gchop.isChecked(), self.gchop_dly.value(), key="gchop")

    def on_dcblock(self):
        if not self.updating and self.cfg:
            self.cfg_call("DC block", self.dev.set_dc_block, self.dcblock.currentData(), self.cfg.dc_block_dis, key="dcblock")

    def on_autostart(self, v):
        if not self.updating and self.cfg:
            self.cfg_call("autostart", self.dev.set_autostart, v, key="autostart")

    def on_apply_trim(self):
        if not self.cfg:
            return
        c = jade.Config.unpack(self.cfg.pack())
        for ch in range(NCH):
            c.phase[ch] = self.adv_phase[ch].value()
            c.ocal[ch] = self.adv_ocal[ch].value()
            c.gcal[ch] = min(0xFFFFFF, int(round(self.adv_gcal[ch].value() * 0x800000)))
        c.dc_block_dis = sum(1 << ch for ch in range(NCH) if not self.adv_dcdis[ch].isChecked())
        self.cfg_call("ADC trim applied", self.dev.set_config, c, key="trim")

    def dev_save(self):
        self.cfg_call("config saved to flash", self.dev.save)

    def dev_load(self):
        self.cfg_call("config loaded from flash", self.dev.load)

    def dev_defaults(self):
        if QMessageBox.question(self, APP_NAME, "Apply firmware default settings (not saved)?") == QMessageBox.Yes:
            self.cfg_call("defaults applied", self.dev.defaults)

    def dev_reset(self):
        self.cfg_call("ADC reset", self.dev.reset_adc)

    def dev_sync(self):
        if self.dev:
            self.worker.run(self.dev.sync, lambda _: self.log("sync"), self.fail("sync"))

    def dev_identify(self):
        if self.dev:
            self.worker.run(self.dev.identify, lambda _: self.log("identify: LEDs chase for 3 s"), self.fail("identify"))

    def dev_reboot(self, boot: bool):
        if not self.dev:
            return
        msg = ("Reboot into the USB bootloader? The board shows up as \"STM32 BOOTLOADER\" (USB DFU) for "
               "STM32CubeProgrammer or dfu-util; press the reset button to leave it." if boot else "Reboot the board?")
        if QMessageBox.question(self, APP_NAME, msg) != QMessageBox.Yes:
            return
        dev = self.dev
        self.worker.run(lambda: dev.reboot(boot), lambda _: self.disconnect_device(), lambda _: self.disconnect_device())

    def on_reg_read(self):
        if not self.dev:
            return
        a = self.reg_addr.value()
        self.worker.run(lambda: self.dev.reg_read(a, 1),
                        lambda v: (self.reg_val.setText(f"{v[0]:04X}"), self.log(f"reg 0x{a:02X} = 0x{v[0]:04X}")),
                        self.fail("register read"))

    def on_reg_write(self):
        if not self.dev:
            return
        a = self.reg_addr.value()
        try:
            val = int(self.reg_val.text(), 16)
        except ValueError:
            QMessageBox.warning(self, APP_NAME, "Value must be hex, e.g. FF0E")
            return
        dev = self.dev

        def job():
            rb = dev.reg_write(a, [val])
            return rb, dev.get_config(), dev.get_telemetry()["gen"]

        def ok(r):
            rb, cfg, gen = r
            self.log(f"reg 0x{a:02X} <- 0x{val:04X}, read back 0x{rb[0]:04X}")
            self.on_config((cfg, gen))
        self.worker.run(job, ok, self.fail("register write"))

    def on_reg_dump(self):
        if not self.dev:
            return

        def ok(v):
            lines = ["  ".join(f"{a + i:02X}:{v[a + i]:04X}" for i in range(8) if a + i < len(v)) for a in range(0, len(v), 8)]
            self.log("registers:\n" + "\n".join(lines))
        self.worker.run(lambda: self.dev.reg_read(0, 0x33), ok, self.fail("register dump"))

    # -------------------------------------------------- channel UI actions
    def refresh_unit_box(self, ch):
        box = self.w_unit[ch]
        s = self.chan[ch]
        box.blockSignals(True)
        box.clear()
        for u in UNITS:
            if u == "cal":
                if s.calibrated:
                    box.addItem(s.cal_unit, "cal")
            else:
                box.addItem(u, u)
        idx = box.findData(s.unit if (s.unit != "cal" or s.calibrated) else "mV")
        box.setCurrentIndex(max(0, idx))
        box.blockSignals(False)

    def on_show(self, ch, v):
        self.chan[ch].show = v
        self.rebuild_plots()

    def on_name(self, ch, text):
        self.chan[ch].name = text.strip() or f"ch{ch}"
        self.update_axis_labels()
        self.refresh_readout_names()
        if self.layout_mode == "overlay":
            self.rebuild_plots()

    def on_unit(self, ch, unit):
        if unit is None:
            return
        self.chan[ch].unit = unit
        self.tare[ch] = 0.0
        self.update_axis_labels()
        self.refresh_readout_names()

    def on_layout(self):
        self.rebuild_plots()

    def open_cal(self, ch):
        dlg = CalDialog(self, ch)
        if dlg.exec():
            self.w_name[ch].setText(self.chan[ch].name)
            self.refresh_unit_box(ch)
            self.tare[ch] = 0.0
            self.rebuild_plots()
            s = self.chan[ch]
            if s.calibrated:
                self.log(f"ch{ch} calibrated: {s.scale_per_mV:.6g} {s.cal_unit}/mV, zero {s.zero_mV:.6f} mV ({s.cal_note})")
                if self.recorder:
                    self.recorder.comment(f"ch{ch} calibration changed: {s.scale_per_mV:.9g} {s.cal_unit}/mV "
                                          f"zero {s.zero_mV:.9g} mV (engineering column keeps the old one)")
            else:
                self.log(f"ch{ch} calibration removed")
            self.save_settings()

    def capture_mv(self, ch: int, seconds: float, done: Callable):
        """Average the raw (unfiltered) mV of a channel over the next `seconds`."""
        t_start = self.store.t[self.store.n - 1] if self.store.n else None

        def finish():
            st = self.store
            if st.n == 0:
                done(0.0, 0.0, 0)
                return
            t_end = st.t[st.n - 1]
            a = int(np.searchsorted(st.t[:st.n], t_start if t_start is not None else t_end - seconds, side="right"))
            y = st.mv[ch, a:st.n]
            y = y[np.isfinite(y)]
            if y.size == 0:
                done(0.0, 0.0, 0)
            else:
                done(float(y.mean()), float(y.std()), int(y.size))
        QTimer.singleShot(int(seconds * 1000), finish)

    def do_tare(self, ch: Optional[int]):
        chans = [ch] if ch is not None else [c for c in range(NCH) if self.chan[c].show]
        st = self.store
        if st.n == 0:
            return
        sl = st.window(0.5)
        for c in chans:
            y = to_display(self.chan[c], st.fmv[c, sl].astype(float), st.gain[c, sl])
            y = y[np.isfinite(y)]
            if y.size:
                self.tare[c] = float(y.mean())
        self.refresh_readout_names()
        self.log("tare " + ", ".join(f"{self.chan[c].name}={fmt(self.tare[c])} {self.chan[c].unit_label()}" for c in chans))

    def clear_tare(self):
        self.tare[:] = 0
        self.refresh_readout_names()

    def set_paused(self, on: bool):
        was = self.paused
        self.paused = bool(on)
        if self.paused and not was:
            # zooming in on a paused view: hand pyqtgraph the full-resolution data
            try:
                self.redraw(full=True)
            except Exception:  # noqa: BLE001
                crash_log("redraw on pause:\n" + traceback.format_exc())
        self.btn_pause.setChecked(self.paused)
        self.btn_pause.setText("Follow (resume)" if self.paused else "Pause view")

    # ------------------------------------------------------------ filters
    def set_filter(self, lp, av):
        self.lp_spin.setValue(lp)
        self.avg_spin.setValue(av)
        self.on_filter()

    def on_filter(self):
        self.lowpass = self.lp_spin.value()
        self.avg = self.avg_spin.value()
        self.rebuild_filters(refilter=True)
        self.update_filter_label()

    def update_filter_label(self):
        parts = []
        if self.lowpass:
            parts.append(f"low-pass {self.lowpass:g} Hz")
        if self.avg > 1:
            parts.append(f"average of {self.avg} samples")
        self.lbl_filter.setText(("Active: " + " + ".join(parts)) if parts else "No filter. Raw data shown.")

    def rebuild_filters(self, refilter: bool):
        rate = self.cfg.data_rate() if self.cfg else 1000.0
        self.filter_rate = rate
        lp = self.lowpass if self.lowpass and self.lowpass < rate / 2 else None
        if self.lowpass and not lp:
            self.log(f"low-pass {self.lowpass:g} Hz is above Nyquist ({rate / 2:g} Hz), ignored", 2)
        self.filters = [jade.LiveFilter(rate, lp, self.avg) for _ in range(NCH)]
        st = self.store
        if refilter and st.n:
            for ch in range(NCH):
                x = st.mv[ch, :st.n].astype(float)
                if np.isfinite(x).any():
                    st.fmv[ch, :st.n] = self.filters[ch](x)
                else:
                    st.fmv[ch, :st.n] = np.nan

    # ---------------------------------------------------------- data path
    def clear_data(self):
        self.store.clear()
        self.t_offset = 0.0
        self._t_last = None
        self.t_origin = None
        self.t_wrap = 0
        self.t_last_raw = None
        self.last_index = None
        self.gaps = 0
        self.markers = []
        self.rebuild_filters(refilter=False)
        self.rebuild_plots()

    def tick(self):
        try:
            self._tick()
        except Exception:  # noqa: BLE001 - never let the timer die
            tb = traceback.format_exc()
            crash_log("tick:\n" + tb)
            now = time.monotonic()
            if now - getattr(self, "_last_tick_err", 0) > 5:
                self._last_tick_err = now
                self.log(f"internal error (details in {CRASH_LOG}): {tb.strip().splitlines()[-1]}", 3)

    def _tick(self):
        if self.dev is None:
            return
        if not self.dev._rx.is_alive():
            self._lose_device(self.dev.error or "reader stopped")
            return
        if self.dev.handler_errors != getattr(self, "_seen_handler_errors", 0):
            self._seen_handler_errors = self.dev.handler_errors
            crash_log("packet handler:\n" + self.dev.last_handler_error)
            self.log(f"bad packet from board ignored ({self.dev.handler_errors} so far)", 2)
        backlog = self.dev.data.qsize()
        if backlog > MAX_BACKLOG:
            dropped = 0
            while self.dev.data.qsize() > MAX_BACKLOG // 4:
                try:
                    self.dev.data.get_nowait()
                    dropped += 1
                except Exception:  # noqa: BLE001
                    break
            self.log(f"PC too slow for this data rate: skipped {dropped} packets on screen "
                     "(recording has a gap). Lower the rate or plot fewer channels.", 2)
        blocks = self.pending
        self.pending = []
        t_end = time.monotonic() + 0.03         # keep the UI responsive
        q = self.dev.data
        while len(blocks) < 4000:
            try:
                blocks.append(q.get_nowait())
            except Exception:  # noqa: BLE001
                break
            if len(blocks) % 64 == 0 and time.monotonic() > t_end:
                break
        new = False
        if blocks:
            self._last_rx = time.monotonic()
            n0 = self.store.n
            self.ingest(blocks)
            new = self.store.n != n0 or bool(blocks)
        now = time.monotonic()
        # Frame rate adapts to the PC: if the event loop comes back late
        # (painting is slow), draw less often instead of falling behind.
        late = now - self._last_tick_at - self.timer.interval() / 1000
        self._last_tick_at = now
        if late > 0.015:
            self._draw_every = min(0.3, self._draw_every * 1.3)
        else:
            self._draw_every = max(0.033, self._draw_every * 0.97)
        if self.paused:
            return
        due = now - self._last_draw >= self._draw_every
        if (new and due) or self._dirty or now - self._last_draw > 0.3:
            self._dirty = False
            self._last_draw = now
            self.redraw()

    def ingest(self, blocks):
        """Convert, filter and store packets. Consecutive packets with the same
        config and channel set are processed as one array (much cheaper than
        packet by packet at high sample rates)."""
        waiting = []
        group: list = []
        gkey = gcfg = None
        now = time.monotonic()
        for b in blocks:
            if waiting:                 # keep packets in order behind one that waits
                waiting.append(b)
                continue
            cfg = self.gen_cfg.get(b.gen)
            if cfg is None:
                if not waiting and not self.pending_since:
                    self.pending_since = now
                if now - self.pending_since < 1.0 or self.cfg is None:
                    waiting.append(b)
                    continue
                cfg = self.cfg          # give up waiting, best effort
            key = (id(cfg), tuple(b.channels))
            if key != gkey and group:
                self._ingest_group(group, gcfg)
                group = []
            gkey, gcfg = key, cfg
            group.append(b)
        if group:
            self._ingest_group(group, gcfg)
        self.pending = waiting[-MAX_BACKLOG:]
        if waiting and not self.resolving and self.dev:
            self.resolving = True
            dev = self.dev

            def job():
                return dev.get_config(), dev.get_telemetry()["gen"]

            def ok(r):
                self.resolving = False
                self.pending_since = 0.0
                self.on_config(r)

            def err(e):
                self.resolving = False
            self.worker.run(job, ok, err)
        if not waiting:
            self.pending_since = 0.0

    def _ingest_group(self, blocks, cfg: jade.Config):
        blocks = [b for b in blocks if len(b.codes)]
        if not blocks:
            return
        rate = cfg.data_rate()
        if abs(rate - self.filter_rate) > 1e-6:
            self.rebuild_filters(refilter=False)
        chans = list(blocks[0].channels)
        ks = np.fromiter((len(b.codes) for b in blocks), np.int64, len(blocks))
        k = int(ks.sum())
        starts = np.concatenate(([0], np.cumsum(ks)[:-1]))
        within = np.arange(k) - np.repeat(starts, ks)

        # device time of each packet, unwrapped across the 32-bit µs rollover
        t0s = np.empty(len(blocks))
        for i, b in enumerate(blocks):
            raw = b.first_t_us
            if self.t_last_raw is not None and raw < self.t_last_raw and self.t_last_raw - raw > 2 ** 31:
                self.t_wrap += 2 ** 32
            self.t_last_raw = raw
            t0s[i] = (raw + self.t_wrap) / 1e6
            if self.last_index is not None and b.first_index != self.last_index + 1:
                self.gaps += 1
            self.last_index = b.first_index + len(b.codes) - 1
        if self.t_origin is None:
            self.t_origin = float(t0s[0])
        t = (np.repeat(t0s, ks) - self.t_origin) + within / rate + self.t_offset
        idx = np.repeat(np.fromiter((b.first_index for b in blocks), np.int64, len(blocks)), ks) + within
        self._rate_n += k
        # Time stamps come from the board clock; the times inside a packet are
        # extrapolated at the nominal rate. Right at a config change (OSR,
        # reset) the last old-rate samples can overlap the first new ones by
        # a sample period or so: squeeze them so time never runs backwards
        # (plots and CSV tools both expect that).
        if len(blocks) > 1:
            t = np.maximum.accumulate(t)
        prev = self._t_last
        if prev is not None and t[0] <= prev:
            nb = int(np.searchsorted(t, prev, side="right"))
            if nb < k:
                t[:nb] = prev + (t[nb] - prev) * np.arange(1, nb + 1) / (nb + 1)
            else:
                t = prev + np.arange(1, k + 1) * 1e-6
        self._t_last = float(t[-1])

        codes = np.concatenate([np.asarray(b.codes) for b in blocks]).astype(float)
        flags = np.concatenate([np.asarray(b.flags) for b in blocks]).astype(np.int64)
        bad = (flags & FLAG_BAD) != 0
        code8 = np.full((NCH, k), np.nan)
        mv = np.full((NCH, k), np.nan)
        gain = np.zeros((NCH, k), np.uint8)
        for j, ch in enumerate(chans):
            g = cfg.gain[ch]
            code8[ch] = codes[:, j]
            mv[ch] = codes[:, j] * (FS_MV / 2 ** 23 / g)
            gain[ch] = g
        if self.recorder:
            self.recorder.write(t, idx, flags, code8, mv)
        mv[:, bad] = np.nan
        fmv = np.full_like(mv, np.nan)
        for ch in chans:
            fmv[ch] = self.filters[ch](mv[ch])
        self.store.append(t, mv, fmv, gain)

    def redraw(self, full: bool = False):
        """Live view: each channel is reduced to min/max per screen column
        before anything else is computed (spikes stay visible, cost stays flat
        whatever the sample rate or window length). Paused view: full data so
        zooming in shows every sample."""
        st = self.store
        if st.n == 0:
            return
        t_end = st.t[st.n - 1]
        sl = st.window(self.window_s)
        x = st.t[sl]
        n = len(x)
        bins = 1500
        step = n // bins if not full else 0
        if step >= 3:
            m = step * bins
            xr = x[:m].reshape(bins, step)
            xd = np.empty(2 * bins)
            xd[0::2] = xr[:, 0]
            xd[1::2] = xr[:, step // 2]
            xd = np.concatenate((xd, x[m:]))
        for ch, c in self.curves.items():
            y = st.fmv[ch, sl]
            g = st.gain[ch, sl]
            if step >= 3:
                yr = y[:m].reshape(bins, step)
                yd = np.empty(2 * bins, np.float32)
                yd[0::2] = np.fmin.reduce(yr, axis=1)
                yd[1::2] = np.fmax.reduce(yr, axis=1)
                gd = np.repeat(g[:m:step], 2)
                y = np.concatenate((yd, y[m:]))
                g = np.concatenate((gd, g[m:]))
                xs = xd
            else:
                xs = x
            c.setData(xs, to_display(self.chan[ch], y.astype(float), g) - self.tare[ch], connect="finite")
        if self.plots and not full:
            first = next(iter(self.plots.values()))
            first.setXRange(t_end - self.window_s, t_end, padding=0)

    def slow_tick(self):
        try:
            self._slow_tick()
        except Exception:  # noqa: BLE001
            crash_log("slow_tick:\n" + traceback.format_exc())

    def _slow_tick(self):
        if (self.dev and self._card_shown() and not self._sd_busy and not self.worker.busy
                and time.monotonic() - self._sd_last_poll > 2.0):
            self.sd_refresh()
        # readouts + stats
        st = self.store
        if st.n:
            span = 1.0 if self.stats_on_1s.isChecked() else self.window_s
            sl = st.window(span)
            if sl.stop - sl.start > 200000:     # very long windows: every n-th sample
                sl = slice(sl.start, sl.stop, (sl.stop - sl.start) // 200000 + 1)
            last = st.window(0.1)
            for ch in range(NCH):
                s = self.chan[ch]
                y = to_display(s, st.fmv[ch, sl].astype(float), st.gain[ch, sl]) - self.tare[ch]
                y = y[np.isfinite(y)]
                ylast = to_display(s, st.fmv[ch, last].astype(float), st.gain[ch, last]) - self.tare[ch]
                ylast = ylast[np.isfinite(ylast)]
                v = float(ylast.mean()) if ylast.size else float("nan")
                if ch in self.readouts:
                    _, lv, ls = self.readouts[ch]
                    lv.setText(fmt(v, 6))
                    if y.size:
                        ls.setText(f"mean {fmt(y.mean())}  σ {fmt(y.std(), 3)}\n"
                                   f"p-p {fmt(y.max() - y.min(), 3)}  min {fmt(y.min())}  max {fmt(y.max())}")
        # status
        now = time.monotonic()
        if now - self._rate_t >= 1.0:
            self.meas_rate = self._rate_n / (now - self._rate_t)
            self._rate_n, self._rate_t = 0, now
        if self.dev:
            tel = self.dev.telemetry
            # the board sends telemetry every second even when not streaming:
            # silence means the firmware stopped (the port itself stays open)
            if tel.get("uptime_ms") != self._last_uptime:
                self._last_uptime = tel.get("uptime_ms")
                self._last_rx = now
            if self.adc_missing and tel.get("adc_ok") and not self.worker.busy:
                self.adc_missing = False
                dev = self.dev

                def job():
                    info = dev.ping()
                    cfg = dev.get_config()
                    gen = dev.get_telemetry()["gen"]
                    dev.stream_start()
                    return info, cfg, gen

                def ok(r):
                    info, cfg, gen = r
                    self.info = info
                    self.info_labels["adc_id"].setText(str(info["adc_id"]))
                    self.on_config((cfg, gen))
                    self.btn_stream.setChecked(True)
                    self.btn_stream.setText("■ Stop")
                    self.log(f"ADS131M08 detected (ID {info['adc_id']}), streaming", 1)

                def err(e):
                    self.adc_missing = True
                self.worker.run(job, ok, err)
            silent = now - self._last_rx
            if silent > 6.0 and self._silent_warned:
                # the board should have rebooted itself by now: reopen the port
                self._lose_device(f"nothing received for {silent:.0f} s")
                return
            if silent > 3.0 and not self._silent_warned and self.cfg:
                self._silent_warned = True
                self.log("BOARD NOT RESPONDING: nothing received for 3 s. It should reboot by itself "
                         "within a few seconds and the app will reconnect. If it doesn't, look at the LEDs: "
                         "red D11 flickering alone = firmware crash; everything frozen = firmware stuck or no "
                         "power (the watchdog should have rebooted it).", 3)
                crash_log(f"board silent for {silent:.1f} s; last telemetry: {tel}")
            elif silent < 1.5 and self._silent_warned:
                self._silent_warned = False
                self.log("board is responding again")
            nominal = self.cfg.data_rate() if self.cfg else 0
            self.st_conn.setText(f"● {self.dev.port or 'sim'}   fw {self.info.get('firmware', '?')}")
            self.st_conn.setStyleSheet("color: #81c784;")
            self.st_rate.setText(f"{nominal:,.0f} SPS nominal · {self.meas_rate:,.0f} received")
            errs = (tel.get("dropped", 0), tel.get("crc_errors", 0), tel.get("missed", 0), self.gaps)
            self.st_err.setText("dropped {} · crc {} · missed {} · gaps {}".format(*errs))
            self.st_err.setStyleSheet("color: #ef5350;" if any(errs) else "")
            if tel and not tel.get("adc_ok", 1):
                self.st_rate.setText("ADC NOT DETECTED")
            if self._silent_warned:
                self.st_conn.setText(f"● {self.dev.port or 'sim'}   BOARD NOT RESPONDING")
                self.st_conn.setStyleSheet("color: #ef5350; font-weight: bold;")
        elif self.reconnecting:
            self.st_conn.setText(f"reconnecting to {self.last_port}…")
            self.st_conn.setStyleSheet("color: #ffb74d; font-weight: bold;")
        else:
            self.st_conn.setStyleSheet("color: #999;")
        if self.recorder:
            el = time.monotonic() - self.recorder.t0
            r = self.recorder
            self.st_rec.setText(f"● REC {int(el // 60):02d}:{el % 60:04.1f}  {r.rows:,} rows  "
                                f"{r.size / 1e6:.1f} MB" + ("  WRITE ERROR" if r.error else
                                                            ("  (disk slow)" if r.backlog > 200 else "")))
            self.st_rec.setStyleSheet("color: #ef5350; font-weight: bold;")
        else:
            self.st_rec.setText("")

    # --------------------------------------------------- record + markers
    def toggle_record(self):
        if self.recorder:
            r, self.recorder = self.recorder, None
            done = r.close()
            self.btn_rec.setText("● Record")
            self.btn_rec.setChecked(False)
            if r.error:
                self.log(f"recording {r.path} had a write error: {r.error.strip().splitlines()[-1]}", 3)
            elif done:
                self.log(f"recording saved: {r.path} ({r.rows:,} rows)")
            else:
                self.log(f"recording {r.path}: {r.rows:,} rows, still being written to disk in the background")
            return
        if not self.cfg:
            self.btn_rec.setChecked(False)
            return
        try:
            Path(self.record_dir).mkdir(parents=True, exist_ok=True)
            path = Path(self.record_dir) / f"jade_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            self.recorder = Recorder(path, self.cfg, self.chan, self.info)
        except OSError as e:
            self.btn_rec.setChecked(False)
            QMessageBox.warning(self, APP_NAME, f"Cannot create the recording:\n{e}")
            return
        self.btn_rec.setText("■ Stop recording")
        self.btn_rec.setChecked(True)
        self.log(f"recording to {path}")

    def add_auto_marker(self, text: str):
        if not self.store.n:
            return
        t = float(self.store.t[self.store.n - 1])
        self.markers.append((t, text))
        self._draw_marker(t, text)

    def add_marker(self):
        if not self.store.n:
            return
        t = float(self.store.t[self.store.n - 1])
        text = self.marker_edit.text().strip() or f"M{len(self.markers) + 1}"
        self.marker_edit.clear()
        self.markers.append((t, text))
        self._draw_marker(t, text)
        if self.recorder:
            self.recorder.comment(f"MARK t={t:.6f} {text}")
        self.log(f"marker '{text}' at t={t:.3f} s")

    def _draw_marker(self, t, text):
        for i, p in enumerate(dict.fromkeys(self.plots.values())):
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen("#ffffff", width=1, style=Qt.DashLine),
                                   label=text if i == 0 else None,
                                   labelOpts={"position": 0.95, "color": "#fff", "fill": (0, 0, 0, 150)})
            p.addItem(line)
            self.marker_lines.append(line)

    # ---------------------------------------------------------- settings
    def settings_dict(self) -> dict:
        return {
            "channels": [asdict(c) for c in self.chan],
            "window_s": self.window_s,
            "layout": "stacked" if self.rb_stacked.isChecked() else "overlay",
            "lowpass": self.lowpass,
            "avg": self.avg,
            "record_dir": self.record_dir,
            "last_port": self.last_port,
        }

    def apply_settings_dict(self, d: dict, ui: bool):
        for i, c in enumerate(d.get("channels", [])[:NCH]):
            known = {k: v for k, v in c.items() if k in ChanSettings.__dataclass_fields__}
            known.setdefault("color", COLORS[i])
            known.setdefault("name", f"ch{i}")
            self.chan[i] = ChanSettings(**known)
        self.window_s = d.get("window_s", self.window_s)
        self.layout_mode = d.get("layout", self.layout_mode)
        self.lowpass = d.get("lowpass", self.lowpass)
        self.avg = d.get("avg", self.avg)
        self.record_dir = d.get("record_dir", self.record_dir)
        self.last_port = d.get("last_port", self.last_port)
        if ui:
            for ch in range(NCH):
                self.w_name[ch].setText(self.chan[ch].name)
                self.w_show[ch].blockSignals(True)
                self.w_show[ch].setChecked(self.chan[ch].show)
                self.w_show[ch].blockSignals(False)
                self.refresh_unit_box(ch)
            self.lp_spin.setValue(self.lowpass)
            self.avg_spin.setValue(self.avg)
            (self.rb_stacked if self.layout_mode == "stacked" else self.rb_overlay).setChecked(True)
            if self.window_s in WINDOWS:
                self.win_box.setCurrentIndex(WINDOWS.index(self.window_s))
            self.on_filter()
            self.rebuild_plots()

    @staticmethod
    def _settings_source() -> Path:
        return SETTINGS_FILE if SETTINGS_FILE.exists() or not OLD_SETTINGS_FILE.exists() else OLD_SETTINGS_FILE

    def load_settings(self):
        try:
            d = json.loads(self._settings_source().read_text())
            self.apply_settings_dict(d, ui=False)
            self._geometry = d.get("geometry")
        except (OSError, ValueError, TypeError):
            pass

    def save_settings(self):
        d = self.settings_dict()
        d["geometry"] = bytes(self.saveGeometry().toBase64()).decode()
        d["state"] = bytes(self.saveState().toBase64()).decode()
        try:
            SETTINGS_FILE.write_text(json.dumps(d, indent=2))
        except OSError as e:
            self.log(f"cannot save settings: {e}", 2)

    def restore_geometry(self):
        try:
            d = json.loads(self._settings_source().read_text())
            if d.get("geometry"):
                self.restoreGeometry(QByteArray.fromBase64(d["geometry"].encode()))
            if d.get("state"):
                self.restoreState(QByteArray.fromBase64(d["state"].encode()))
        except (OSError, ValueError):
            pass

    def export_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export profile", "jade_profile.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.settings_dict(), indent=2))
            self.log(f"profile exported to {path}")

    def import_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import profile", "", "JSON (*.json)")
        if path:
            try:
                self.apply_settings_dict(json.loads(Path(path).read_text()), ui=True)
                self.log(f"profile loaded from {path}")
            except (OSError, ValueError, TypeError) as e:
                QMessageBox.warning(self, APP_NAME, f"Cannot load profile:\n{e}")

    def import_cli_cal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import cal.json", "", "JSON (*.json)")
        if not path:
            return
        try:
            cal = jade.load_cal(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, APP_NAME, f"Cannot read {path}:\n{e}")
            return
        for ch, c in cal.items():
            if 0 <= ch < NCH:
                s = self.chan[ch]
                s.name, s.cal_unit = c["name"], c["unit"]
                s.scale_per_mV, s.zero_mV, s.unit = c["scale_per_mV"], c["zero_mV"], "cal"
                s.cal_note = "imported"
                self.w_name[ch].setText(s.name)
                self.refresh_unit_box(ch)
        self.rebuild_plots()
        self.log(f"calibration imported for channels {sorted(cal)}")

    def choose_record_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Recording folder", self.record_dir)
        if d:
            self.record_dir = d
            self.log(f"recordings go to {d}")

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plot image", f"jade_{time.strftime('%Y%m%d_%H%M%S')}.png",
                                              "PNG (*.png)")
        if path:
            self.glw.grab().save(path)
            self.log(f"image saved: {path}")

    def closeEvent(self, ev):
        if self.recorder:
            self.recorder.close()
            self.recorder = None
        self.save_settings()
        if self.dev:
            try:
                self.dev.command(jade.CMD["STREAM_STOP"], timeout=0.5)
            except Exception:  # noqa: BLE001
                pass
            self.dev.close()
            self.dev = None
        super().closeEvent(ev)


# ==========================================================================
def dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text = QColor(30, 30, 30), QColor(45, 45, 45), QColor(220, 220, 220)
    p.setColor(QPalette.Window, alt)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.ToolTipBase, alt)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, alt)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.Highlight, QColor(79, 195, 247))
    p.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(110, 110, 110))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(110, 110, 110))
    app.setPalette(p)
    app.setStyleSheet("QPushButton:checked { background: #1565c0; }")


def main():
    # hard crashes (Qt / native) leave a Python stack trace in the crash log
    try:
        _fh = open(CRASH_LOG, "a", encoding="utf-8")
        _fh.write(f"\n----- start {time.strftime('%Y-%m-%d %H:%M:%S')} -----\n")
        _fh.flush()
        faulthandler.enable(_fh, all_threads=True)
    except OSError:
        pass
    pg.setConfigOptions(antialias=False, background="#1a1a1a", foreground="#cccccc")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    dark_palette(app)
    w = MainWindow()
    w.restore_geometry()
    w.show()

    def excepthook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        sys.stderr.write(text) if sys.stderr else None
        crash_log(text)
        try:
            w.log(f"internal error (details in {CRASH_LOG}): {v}", 3)
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = excepthook
    threading.excepthook = lambda a: excepthook(a.exc_type, a.exc_value, a.exc_traceback)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
