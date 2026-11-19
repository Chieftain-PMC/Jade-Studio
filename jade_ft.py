"""
Force / torque view for a 6-gage F/T sensor (ATI Mini58 on Jade).

Six Jade channels carry the six strain-gage signals. Forces and torques are

    [Fx Fy Fz Tx Ty Tz] = C . (V - V0)        V in mV at the ADC pins, as Jade Studio stores them

C is built here from load cases (a known mass hung or placed in a known
orientation, captured as "zero" then "loaded"), or imported from
mini58_cal.py (fit --out file). Each row of C is fitted on its own, using only
the cases where that component of the load is known, so a case with an
unknown hook position still teaches Fx/Fy/Fz without pretending to know Tx/Ty.

Axes with no case that moves them are shown as "not calibrated".
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView,
)

AXES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
UNITS = ["N", "N", "N", "N·m", "N·m", "N·m"]
AX_COLORS = ["#e57373", "#81c784", "#4fc3f7", "#ffb74d", "#ba68c8", "#fff176"]
G = 9.80665
DIRS = {   # gravity direction expressed in sensor axes
    "-z": ("Flat, tool face up (gravity −Z)", (0, 0, -1)),
    "+z": ("Flat, tool face down (gravity +Z)", (0, 0, 1)),
    "+x": ("On its side, +X pointing down", (1, 0, 0)),
    "-x": ("On its side, +X pointing up", (-1, 0, 0)),
    "+y": ("On its side, +Y pointing down", (0, 1, 0)),
    "-y": ("On its side, +Y pointing up", (0, -1, 0)),
}
TSVD_REL = 0.05     # singular values below 5 % of the largest are treated as noise

# Default load cases, all CH1..CH6 with the soldered 1.25 V divider, dV in mV at the ADC pins.
#   Fz: 3112 g centred twice (JADE0006), 554 g at six spots (JADE0007), sensor flat.
#   Fx: 1135 g, four +X and one -X (JADE0009, 2026-11-18).
#   Fy: 1135 g, both directions (JADE0011 and the t.csv plot export, 2026-11-18). PROVISIONAL:
#       the two directions disagree (part of the weight was probably supported in one of them).
# No torque case yet: Tx, Ty, Tz need loads at a measured position (see the dialog).
# Valid only while the wiring, divider and channel signs stay the same.
DEFAULT_CASES = [
    {"note": "JADE0006 3112 g #1 (seed)", "mass_g": 3112, "dir": "-z", "pos": None,
     "dV": [-0.02400, 0.17520, -0.02190, 0.22180, -0.04260, 0.00140]},
    {"note": "JADE0006 3112 g #2 (seed)", "mass_g": 3112, "dir": "-z", "pos": None,
     "dV": [-0.02260, 0.17440, -0.02050, 0.19310, -0.03580, -0.00110]},
    {"note": "JADE0007 554 g S3 (seed)", "mass_g": 554, "dir": "-z", "pos": None,
     "dV": [-0.00236, 0.02055, -0.00023, 0.02529, -0.03070, -0.00116]},
    {"note": "JADE0007 554 g S5 (seed)", "mass_g": 554, "dir": "-z", "pos": None,
     "dV": [-0.00144, 0.01769, 0.00186, 0.01943, -0.03713, -0.00159]},
    {"note": "JADE0007 554 g S6 (seed)", "mass_g": 554, "dir": "-z", "pos": None,
     "dV": [-0.00066, 0.03482, -0.00106, 0.04660, 0.01284, -0.00262]},
    {"note": "JADE0007 554 g S7 (seed)", "mass_g": 554, "dir": "-z", "pos": None,
     "dV": [-0.00029, 0.00339, 0.00396, 0.01303, -0.05259, -0.00373]},
    {"note": "JADE0007 554 g S8 (seed)", "mass_g": 554, "dir": "-z", "pos": None,
     "dV": [-0.00516, -0.01450, 0.00084, 0.05639, -0.03039, -0.00366]},
    {"note": "JADE0007 554 g S9 (seed)", "mass_g": 554, "dir": "-z", "pos": None,
     "dV": [-0.00363, 0.06215, -0.00105, -0.00981, -0.01911, 0.00034]},
    {"note": "JADE0009 1135 g X+ #1", "mass_g": 1135, "dir": "+x", "pos": None,
     "dV": [0.12932, -0.13009, 0.01132, 0.06612, -0.07859, 0.09340]},
    {"note": "JADE0009 1135 g X+ #2", "mass_g": 1135, "dir": "+x", "pos": None,
     "dV": [0.11002, -0.12214, -0.00478, 0.06974, -0.07970, 0.11308]},
    {"note": "JADE0009 1135 g X- #1", "mass_g": 1135, "dir": "-x", "pos": None,
     "dV": [-0.12154, 0.12511, -0.03493, -0.04570, 0.03287, -0.11902]},
    {"note": "JADE0009 1135 g X+ #3", "mass_g": 1135, "dir": "+x", "pos": None,
     "dV": [0.13970, -0.11920, 0.01369, 0.05390, -0.07181, 0.07818]},
    {"note": "JADE0009 1135 g X+ #4", "mass_g": 1135, "dir": "+x", "pos": None,
     "dV": [0.12225, -0.12170, 0.01027, 0.05653, -0.06478, 0.09588]},
    {"note": "JADE0011 1135 g Y+ A1 (provisional)", "mass_g": 1135, "dir": "+y", "pos": None,
     "dV": [0.03851, -0.03367, -0.06428, -0.08969, -0.07816, 0.01616]},
    {"note": "JADE0011 1135 g Y+ A2 (provisional)", "mass_g": 1135, "dir": "+y", "pos": None,
     "dV": [0.02201, -0.02242, -0.03998, -0.08644, -0.07895, 0.02267]},
    {"note": "JADE0011 1135 g Y+ A3 (provisional)", "mass_g": 1135, "dir": "+y", "pos": None,
     "dV": [0.01501, 0.00161, -0.01222, -0.07007, -0.07159, -0.00892]},
    {"note": "JADE0011 1135 g Y- B1 (provisional)", "mass_g": 1135, "dir": "-y", "pos": None,
     "dV": [0.00523, 0.02340, 0.15098, 0.12025, 0.09899, 0.01813]},
    {"note": "JADE0011 1135 g Y- B2 (provisional)", "mass_g": 1135, "dir": "-y", "pos": None,
     "dV": [-0.01250, -0.00756, 0.08897, 0.09862, 0.09357, 0.07463]},
    {"note": "JADE0011 1135 g Y- B3 (provisional)", "mass_g": 1135, "dir": "-y", "pos": None,
     "dV": [0.00458, -0.00092, 0.10546, 0.10294, 0.10069, 0.04552]},
    {"note": "t.csv 1135 g Y+ P1a (provisional)", "mass_g": 1135, "dir": "+y", "pos": None,
     "dV": [0.06580, -0.02500, -0.14000, -0.12280, -0.13380, 0.00260]},
    {"note": "t.csv 1135 g Y+ P1b (provisional)", "mass_g": 1135, "dir": "+y", "pos": None,
     "dV": [0.07070, -0.01610, -0.13770, -0.11190, -0.13435, -0.01530]},
    {"note": "t.csv 1135 g Y- P2a (provisional)", "mass_g": 1135, "dir": "-y", "pos": None,
     "dV": [-0.00815, 0.00230, 0.05190, 0.07300, 0.07675, 0.02690]},
]
OLD_SEED_NOTES = ["JADE0006 3112 g #1 (seed)", "JADE0006 3112 g #2 (seed)", "JADE0007 554 g S3 (seed)", "JADE0007 554 g S5 (seed)", "JADE0007 554 g S6 (seed)", "JADE0007 554 g S7 (seed)", "JADE0007 554 g S8 (seed)", "JADE0007 554 g S9 (seed)"]
SEED_CASES = DEFAULT_CASES   # backward name


def load_vector(case: dict) -> np.ndarray:
    """Known load of a case: forces from mass and orientation, torques from the hook
    position when known (NaN when not, except the component along gravity, always 0)."""
    u = np.array(DIRS[case["dir"]][1], float)
    F = case["mass_g"] / 1000.0 * G * u
    if case.get("pos") is not None:
        T = np.cross(np.asarray(case["pos"], float) / 1000.0, F)
    else:
        T = np.full(3, np.nan)
        T[int(np.argmax(np.abs(u)))] = 0.0
    return np.concatenate([F, T])


def tsvd_solve(D: np.ndarray, y: np.ndarray, rel: float = TSVD_REL) -> np.ndarray:
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    keep = s >= rel * s[0] if s.size and s[0] > 0 else np.zeros_like(s, bool)
    inv = np.where(keep, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    return Vt.T @ (inv * (U.T @ y))


class FTModel:
    def __init__(self):
        self.gage_ch: List[int] = [1, 2, 3, 4, 5, 6]
        self.exc_ch: Optional[int] = None
        self.exc_ratio = 3.0
        self.exc_nominal = 2.5
        self.cases: List[dict] = [dict(c) for c in DEFAULT_CASES]
        self.upgraded = False
        self.V0: Optional[List[float]] = None
        self.imported: Optional[dict] = None      # {"C": 6x6 N per mV, "note": str}
        self.C = np.zeros((6, 6))
        self.status = [""] * 6
        self.ok = np.zeros(6, bool)
        self.solve()

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"gage_ch": self.gage_ch, "exc_ch": self.exc_ch, "exc_ratio": self.exc_ratio,
                "exc_nominal": self.exc_nominal, "cases": self.cases, "V0": self.V0,
                "imported": self.imported}

    def from_dict(self, d: dict):
        if not d:
            return
        self.gage_ch = list(d.get("gage_ch", self.gage_ch))[:6]
        self.exc_ch = d.get("exc_ch", self.exc_ch)
        self.exc_ratio = float(d.get("exc_ratio", self.exc_ratio))
        self.exc_nominal = float(d.get("exc_nominal", self.exc_nominal))
        if "cases" in d:
            self.cases = list(d["cases"])
            notes = [c.get("note") for c in self.cases]
            if notes == OLD_SEED_NOTES:          # untouched 14 Nov default -> newer default set
                self.cases = [dict(c) for c in DEFAULT_CASES]
                self.upgraded = True
        self.V0 = d.get("V0", self.V0)
        self.imported = d.get("imported", self.imported)
        self.solve()

    # -------------------------------------------------------------- solving
    def solve(self):
        self.provisional = np.zeros(6, bool)
        if self.imported:
            self.C = np.asarray(self.imported["C"], float)
            self.ok = np.any(self.C != 0, axis=1)
            self.status = [f"imported ({self.imported.get('note', 'file')})" if o else "not in file" for o in self.ok]
            return
        self.C = np.zeros((6, 6))
        self.ok = np.zeros(6, bool)
        self.provisional = np.zeros(6, bool)
        self.status = ["no load case moves this axis"] * 6
        if not self.cases:
            return
        D = np.array([c["dV"] for c in self.cases], float)
        L = np.array([load_vector(c) for c in self.cases])
        for i in range(6):
            m = np.isfinite(L[:, i])
            y = L[m, i]
            if m.sum() == 0 or np.max(np.abs(y)) < 1e-9:
                continue
            c = tsvd_solve(D[m], y)
            self.C[i] = c
            self.ok[i] = True
            res = D[m] @ c - y
            moved = int(np.sum(np.abs(y) > 1e-9))
            prov = any("provisional" in str(c.get("note", "")).lower() and abs(Lc[i]) > 1e-9
                       for c, Lc in zip(self.cases, L))
            self.provisional[i] = prov
            self.status[i] = (("PROVISIONAL: " if prov else "") + f"{moved} case(s) with load, {int(m.sum())} used, "
                              f"residual rms {np.sqrt(np.mean(res ** 2)):.3g} {UNITS[i]}")

    def residuals(self) -> np.ndarray:
        """Per case and axis: fitted minus known load (NaN where unknown)."""
        if not self.cases:
            return np.zeros((0, 6))
        D = np.array([c["dV"] for c in self.cases], float)
        L = np.array([load_vector(c) for c in self.cases])
        return D @ self.C.T - L

    # ------------------------------------------------------------- data
    def gage_mv(self, mv_all: np.ndarray) -> np.ndarray:
        """(8, n) mV for all channels -> (6, n) gage mV, excitation-corrected if set."""
        X = mv_all[self.gage_ch].astype(float)
        if self.exc_ch is not None:
            vexc = mv_all[self.exc_ch].astype(float) * self.exc_ratio / 1000.0
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X * np.where(vexc > 0.1 * self.exc_nominal, self.exc_nominal / vexc, np.nan)
        return X

    def forces(self, X: np.ndarray) -> np.ndarray:
        V0 = np.asarray(self.V0, float) if self.V0 is not None else np.nanmean(X, axis=1)
        return self.C @ (X - V0[:, None])


# ==========================================================================
class FTPanel(QWidget):
    def __init__(self, main, settings: Optional[dict] = None):
        super().__init__()
        self.main = main
        self.model = FTModel()
        self.model.from_dict(settings or {})
        self.trail_s = 1.0
        self.xy_range = 2.0
        self.bar_range = [2.0, 0.2]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        b = QPushButton("Tare F/T")
        b.setToolTip("Zero all six axes on the last 0.5 s (also done by 'Tare all' / T)")
        b.clicked.connect(lambda: self.tare())
        row.addWidget(b)
        b = QPushButton("Calibration…")
        b.clicked.connect(self.open_cal)
        row.addWidget(b)
        self.cb_time = QCheckBox("Time plot")
        self.cb_time.setChecked(True)
        self.cb_time.toggled.connect(lambda v: self.time_area.setVisible(v))
        row.addWidget(self.cb_time)
        row.addStretch()
        lay.addLayout(row)

        grid = QGridLayout()
        self.val = []
        self.names = []
        for i, ax in enumerate(AXES):
            name = QLabel(ax)
            self.names.append(name)
            name.setStyleSheet(f"color: {AX_COLORS[i]}; font-weight: bold;")
            v = QLabel("—")
            v.setFont(QFont("Consolas", 14, QFont.Bold))
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            u = QLabel(UNITS[i])
            r, c = i % 3, (i // 3) * 3
            grid.addWidget(name, r, c)
            grid.addWidget(v, r, c + 1)
            grid.addWidget(u, r, c + 2)
            self.val.append(v)
        lay.addLayout(grid)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setMinimumHeight(260)
        lay.addWidget(self.glw, 3)
        self.xy = self.glw.addPlot(row=0, col=0, title="Fx / Fy (N)")
        self.xy.setAspectLocked(True)
        self.xy.showGrid(x=True, y=True, alpha=0.25)
        self.xy.setLabel("bottom", "Fx")
        self.xy.setLabel("left", "Fy")
        self.xy.addLine(x=0, pen=pg.mkPen("#666"))
        self.xy.addLine(y=0, pen=pg.mkPen("#666"))
        self.xy_trail = self.xy.plot(pen=pg.mkPen("#4fc3f7", width=1))
        self.xy_dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush("#4fc3f7"), pen=pg.mkPen("w"))
        self.xy.addItem(self.xy_dot)
        self.xy.setMouseEnabled(False, False)
        self.xy_note = pg.TextItem("", color="#aaa", anchor=(0.5, 0.5))
        self.xy.addItem(self.xy_note)

        self.barF = self.glw.addPlot(row=0, col=1, title="Forces (N)")
        self.barT = self.glw.addPlot(row=0, col=2, title="Torques (N·m)")
        for p, names in ((self.barF, AXES[:3]), (self.barT, AXES[3:])):
            p.showGrid(y=True, alpha=0.25)
            p.getAxis("bottom").setTicks([[(k, n) for k, n in enumerate(names)]])
            p.addLine(y=0, pen=pg.mkPen("#888"))
            p.setMouseEnabled(False, False)
            p.setXRange(-0.6, 2.6, padding=0)
        self.bF = pg.BarGraphItem(x=[0, 1, 2], height=[0, 0, 0], width=0.6, brushes=AX_COLORS[:3])
        self.bT = pg.BarGraphItem(x=[0, 1, 2], height=[0, 0, 0], width=0.6, brushes=AX_COLORS[3:])
        self.barF.addItem(self.bF)
        self.barT.addItem(self.bT)
        self.glw.ci.layout.setColumnStretchFactor(0, 3)
        self.glw.ci.layout.setColumnStretchFactor(1, 2)
        self.glw.ci.layout.setColumnStretchFactor(2, 2)

        self.time_area = pg.GraphicsLayoutWidget()
        self.time_area.setMinimumHeight(200)
        lay.addWidget(self.time_area, 2)
        self.tF = self.time_area.addPlot(row=0, col=0)
        self.tT = self.time_area.addPlot(row=1, col=0)
        self.tT.setXLink(self.tF)
        self.tF.setLabel("left", "N")
        self.tT.setLabel("left", "N·m")
        self.tT.setLabel("bottom", "time", units="s")
        self.tF.addLegend(offset=(5, 5))
        self.tT.addLegend(offset=(5, 5))
        self.curves = []
        for i, ax in enumerate(AXES):
            p = self.tF if i < 3 else self.tT
            p.showGrid(x=True, y=True, alpha=0.25)
            self.curves.append(p.plot(pen=pg.mkPen(AX_COLORS[i], width=1), name=ax))

        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color: #999;")
        lay.addWidget(self.lbl)
        self.refresh_status()
        if getattr(self.model, "upgraded", False):
            QTimer.singleShot(500, lambda: self.main.log("F/T: default load cases updated (Fx, Fz, provisional Fy)"))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(50)

    # ----------------------------------------------------------- helpers
    def to_dict(self) -> dict:
        return self.model.to_dict()

    def from_dict(self, d: dict):
        self.model.from_dict(d)
        self.refresh_status()

    def changed(self):
        self.model.solve()
        self.refresh_status()
        try:
            self.main.save_settings()
        except Exception:  # noqa: BLE001
            pass

    def refresh_status(self):
        m = self.model
        cal = [ax for ax, ok in zip(AXES, m.ok) if ok]
        txt = ("Calibrated: " + ", ".join(cal)) if cal else "No axis calibrated yet: Calibration… → add load cases."
        prov = [ax for ax, p in zip(AXES, m.provisional) if p]
        if prov:
            txt += "   * provisional: " + ", ".join(prov)
        missing = [ax for ax, ok in zip(AXES, m.ok) if not ok]
        if missing:
            txt += "   Not calibrated: " + ", ".join(missing)
        if m.V0 is None:
            txt += "   Not tared yet (press Tare F/T with no load)."
        txt += f"   Gages on ch{', ch'.join(map(str, m.gage_ch))}"
        if m.exc_ch is not None:
            txt += f", excitation ch{m.exc_ch} ×{m.exc_ratio:g} (nominal {m.exc_nominal:g} V)"
        self.lbl.setText(txt)
        for i in range(6):
            self.val[i].setEnabled(bool(m.ok[i]))
            self.val[i].setToolTip(m.status[i])
            self.names[i].setText(AXES[i] + ("*" if m.provisional[i] else ""))
            self.names[i].setToolTip(m.status[i])

    def tare(self, quiet: bool = False):
        st = self.main.store
        if st.n == 0:
            return
        sl = st.window(0.5)
        X = self.model.gage_mv(st.mv[:, sl])
        if not np.isfinite(X).any():
            return
        self.model.V0 = np.nanmean(X, axis=1).tolist()
        self.refresh_status()
        if not quiet:
            self.main.log("F/T tared")
        try:
            self.main.save_settings()
        except Exception:  # noqa: BLE001
            pass

    def capture(self, seconds: float, done: Callable):
        """Average the six gage channels (raw mV) over the next `seconds`."""
        st = self.main.store
        t_start = st.t[st.n - 1] if st.n else None

        def finish():
            s = self.main.store
            if s.n == 0 or t_start is None:
                done(None, None, 0)
                return
            a = int(np.searchsorted(s.t[:s.n], t_start, side="right"))
            X = self.model.gage_mv(s.mv[:, a:s.n])
            ok = np.all(np.isfinite(X), axis=0)
            if ok.sum() == 0:
                done(None, None, 0)
                return
            done(X[:, ok].mean(axis=1), X[:, ok].std(axis=1), int(ok.sum()))
        QTimer.singleShot(int(seconds * 1000), finish)

    def open_cal(self):
        FTCalDialog(self).exec()

    # ------------------------------------------------------------ drawing
    def update_view(self):
        if not self.isVisible():
            return
        st = self.main.store
        if st.n == 0 or not self.model.ok.any():
            for v in self.val:
                v.setText("—")
            return
        m = self.model
        sl = st.window(self.main.window_s)
        n = sl.stop - sl.start
        step = max(1, n // 3000)
        sl2 = slice(sl.start, sl.stop, step)
        X = m.gage_mv(st.fmv[:, sl2])
        if not np.isfinite(X).any():
            return
        F = m.forces(X)
        t = st.t[sl2]
        last = st.window(0.1)
        Fl = m.forces(m.gage_mv(st.fmv[:, last]))
        now = np.nanmean(Fl, axis=1) if Fl.size else np.full(6, np.nan)
        for i in range(6):
            self.val[i].setText(f"{now[i]:+.3f}" if m.ok[i] and np.isfinite(now[i]) else "—")
        # XY
        if m.ok[0] or m.ok[1]:
            fx = F[0] if m.ok[0] else np.zeros_like(F[0])
            fy = F[1] if m.ok[1] else np.zeros_like(F[1])
            k = t >= t[-1] - self.trail_s
            self.xy_trail.setData(fx[k], fy[k], connect="finite")
            px = now[0] if m.ok[0] else 0.0
            py = now[1] if m.ok[1] else 0.0
            self.xy_dot.setData([px], [py])
            r = max(abs(px), abs(py), 1.0) * 1.3
            self.xy_range = max(r, self.xy_range * 0.98)
            self.xy.setXRange(-self.xy_range, self.xy_range, padding=0)
            self.xy.setYRange(-self.xy_range, self.xy_range, padding=0)
            self.xy_note.setText("" if (m.ok[0] and m.ok[1]) else
                                 ("Fy not calibrated" if m.ok[0] else "Fx not calibrated"))
            self.xy_note.setPos(0, self.xy_range * 0.85)
        else:
            self.xy_dot.setData([], [])
            self.xy_trail.setData([], [])
            self.xy_note.setText("Fx / Fy not calibrated")
            self.xy_note.setPos(0, 0)
            self.xy.setXRange(-1, 1, padding=0)
            self.xy.setYRange(-1, 1, padding=0)
        # bars
        for bar, plot, idx, j in ((self.bF, self.barF, [0, 1, 2], 0), (self.bT, self.barT, [3, 4, 5], 1)):
            h = [now[i] if m.ok[i] and np.isfinite(now[i]) else 0.0 for i in idx]
            bar.setOpts(height=h)
            r = max(max(abs(x) for x in h) * 1.3, 0.5 if j == 0 else 0.05)
            self.bar_range[j] = max(r, self.bar_range[j] * 0.98)
            plot.setYRange(-self.bar_range[j], self.bar_range[j], padding=0)
        # time
        if self.time_area.isVisible():
            for i in range(6):
                if m.ok[i]:
                    self.curves[i].setData(t, F[i], connect="finite")
                else:
                    self.curves[i].setData([], [])
            self.tT.setVisible(bool(m.ok[3:].any()))
            self.tF.setXRange(t[-1] - self.main.window_s, t[-1], padding=0)


# ==========================================================================
class FTCalDialog(QDialog):
    def __init__(self, panel: FTPanel):
        super().__init__(panel)
        self.panel, self.model = panel, panel.model
        self.setWindowTitle("Force / torque calibration")
        self.resize(980, 760)
        self.zero = None
        self.loaded = None
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Each load case = a known mass in a known orientation. Capture zero with nothing on the sensor, "
            "put the mass on (or hang it), wait until it is still, capture loaded, then Add case. "
            "Axes are in the sensor's own frame (marked on the body). A weight sitting on the tool face "
            "with the sensor flat reads Fz < 0. Enter the hook position if you know it: it calibrates "
            "the torques too. Each axis needs cases that move it: sensor on its side with ±X down for Fx, "
            "±Y down for Fy.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        top = QHBoxLayout()
        g = QGroupBox("New load case")
        f = QFormLayout(g)
        self.mass = QDoubleSpinBox()
        self.mass.setRange(0, 1e6)
        self.mass.setDecimals(1)
        self.mass.setSuffix(" g")
        self.mass.setValue(500)
        f.addRow("Mass (measured)", self.mass)
        self.dir = QComboBox()
        for k, (label, _) in DIRS.items():
            self.dir.addItem(label, k)
        f.addRow("Orientation", self.dir)
        self.pos_known = QCheckBox("Hook / contact point known")
        f.addRow(self.pos_known)
        pr = QHBoxLayout()
        self.px, self.py, self.pz = [QDoubleSpinBox() for _ in range(3)]
        for s, n in ((self.px, "x"), (self.py, "y"), (self.pz, "z")):
            s.setRange(-1000, 1000)
            s.setDecimals(1)
            s.setSuffix(" mm")
            s.setEnabled(False)
            pr.addWidget(QLabel(n))
            pr.addWidget(s)
        self.pos_known.toggled.connect(lambda v: [s.setEnabled(v) for s in (self.px, self.py, self.pz)])
        f.addRow("Position from sensor origin", pr)
        self.avg = QDoubleSpinBox()
        self.avg.setRange(0.5, 30)
        self.avg.setValue(3.0)
        self.avg.setSuffix(" s")
        f.addRow("Average over", self.avg)
        self.note = QLineEdit()
        self.note.setPlaceholderText("note (optional)")
        f.addRow("Note", self.note)
        self.btn_zero = QPushButton("1. Capture zero (no load)")
        self.lbl_zero = QLabel("—")
        f.addRow(self.btn_zero, self.lbl_zero)
        self.btn_load = QPushButton("2. Capture loaded")
        self.lbl_load = QLabel("—")
        f.addRow(self.btn_load, self.lbl_load)
        self.btn_add = QPushButton("3. Add case")
        self.btn_add.setEnabled(False)
        self.lbl_case = QLabel("")
        self.lbl_case.setWordWrap(True)
        f.addRow(self.btn_add, self.lbl_case)
        top.addWidget(g, 3)

        g = QGroupBox("Channels")
        f = QFormLayout(g)
        self.gage = QLineEdit(",".join(map(str, self.model.gage_ch)))
        f.addRow("Gage channels (SG0..SG5)", self.gage)
        self.exc = QComboBox()
        self.exc.addItem("none", None)
        for c in range(8):
            self.exc.addItem(f"ch{c}", c)
        self.exc.setCurrentIndex(0 if self.model.exc_ch is None else self.model.exc_ch + 1)
        f.addRow("Excitation monitor", self.exc)
        self.exc_ratio = QDoubleSpinBox()
        self.exc_ratio.setRange(0.1, 100)
        self.exc_ratio.setDecimals(4)
        self.exc_ratio.setValue(self.model.exc_ratio)
        f.addRow("Excitation = ratio × channel", self.exc_ratio)
        self.exc_nom = QDoubleSpinBox()
        self.exc_nom.setRange(0.1, 30)
        self.exc_nom.setDecimals(3)
        self.exc_nom.setSuffix(" V")
        self.exc_nom.setValue(self.model.exc_nominal)
        f.addRow("Nominal excitation", self.exc_nom)
        b = QPushButton("Apply channel settings")
        b.clicked.connect(self.apply_channels)
        f.addRow(b)
        warn = QLabel("Changing channels or the excitation setting makes existing cases invalid.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #ffb74d;")
        f.addRow(warn)
        top.addWidget(g, 2)
        lay.addLayout(top)

        cols = ["Note", "Mass g", "Orientation", "Position mm"] + [f"{a} known" for a in AXES] + [f"{a} fit−known" for a in AXES]
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lay.addWidget(self.table, 2)

        row = QHBoxLayout()
        for txt, fn in (("Delete selected", self.delete_selected), ("Load default cases", self.add_seed),
                        ("Export cases…", self.export_cases), ("Import cases…", self.import_cases),
                        ("Import matrix (mini58_cal fit)…", self.import_matrix),
                        ("Use cases (drop imported matrix)", self.drop_matrix)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)

        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setFont(QFont("Consolas", 9))
        self.result.setMaximumHeight(170)
        lay.addWidget(self.result)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)

        self.btn_zero.clicked.connect(lambda: self.capture("zero"))
        self.btn_load.clicked.connect(lambda: self.capture("load"))
        self.btn_add.clicked.connect(self.add_case)
        self.refresh()

    # -------------------------------------------------------------- capture
    def capture(self, what: str):
        sec = self.avg.value()
        for b in (self.btn_zero, self.btn_load, self.btn_add):
            b.setEnabled(False)
        lbl = self.lbl_zero if what == "zero" else self.lbl_load
        lbl.setText(f"averaging {sec:g} s…")

        def done(mean, sd, n):
            self.btn_zero.setEnabled(True)
            self.btn_load.setEnabled(True)
            if n == 0:
                lbl.setText("no data: are the gage channels enabled and streaming?")
            else:
                lbl.setText(f"{n} samples, worst σ {np.max(sd) * 1e3:.2f} µV")
                if what == "zero":
                    self.zero = mean
                else:
                    self.loaded = mean
            ready = self.zero is not None and self.loaded is not None
            self.btn_add.setEnabled(ready)
            if ready:
                dv = (self.loaded - self.zero) * 1e3
                self.lbl_case.setText("ΔV (µV): " + "  ".join(f"{x:+.1f}" for x in dv))
        self.panel.capture(sec, done)

    def current_case(self) -> dict:
        pos = [self.px.value(), self.py.value(), self.pz.value()] if self.pos_known.isChecked() else None
        return {"note": self.note.text().strip() or time.strftime("%H:%M:%S"),
                "mass_g": self.mass.value(), "dir": self.dir.currentData(), "pos": pos,
                "dV": (self.loaded - self.zero).tolist()}

    def add_case(self):
        if self.zero is None or self.loaded is None:
            return
        c = self.current_case()
        if c["mass_g"] <= 0:
            QMessageBox.warning(self, "F/T", "Enter the mass first.")
            return
        self.model.imported = None
        self.model.cases.append(c)
        self.loaded = None
        self.btn_add.setEnabled(False)
        self.lbl_load.setText("—")
        self.lbl_case.setText("added")
        self.panel.changed()
        self.refresh()

    # ---------------------------------------------------------------- table
    def refresh(self):
        m = self.model
        L = np.array([load_vector(c) for c in m.cases]) if m.cases else np.zeros((0, 6))
        R = m.residuals()
        self.table.setRowCount(len(m.cases))
        for r, c in enumerate(m.cases):
            pos = "unknown" if c.get("pos") is None else ", ".join(f"{v:g}" for v in c["pos"])
            vals = [c.get("note", ""), f"{c['mass_g']:g}", DIRS[c["dir"]][0], pos]
            vals += ["?" if not np.isfinite(x) else f"{x:+.3f}" for x in L[r]]
            vals += ["" if not np.isfinite(x) or not m.ok[i] else f"{x:+.3f}" for i, x in enumerate(R[r])]
            for k, v in enumerate(vals):
                self.table.setItem(r, k, QTableWidgetItem(v))
        lines = []
        if m.imported:
            lines.append(f"Using imported matrix: {m.imported.get('note', '')}")
        for i, ax in enumerate(AXES):
            row = "  ".join(f"{x:+9.4g}" for x in m.C[i]) if m.ok[i] else ""
            lines.append(f"{ax}: {m.status[i]}")
            if row:
                lines.append(f"    {ax} [{UNITS[i]}] = Σ c·dV(mV): {row}")
        self.result.setPlainText("\n".join(lines))

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.model.cases[r]
        if rows:
            self.panel.changed()
            self.refresh()

    def add_seed(self):
        if self.model.cases and QMessageBox.question(
                self, "F/T", "Replace the current load cases with the default set (Fx, Fz, provisional Fy)?"
        ) != QMessageBox.Yes:
            return
        self.model.imported = None
        self.model.cases = [dict(c) for c in DEFAULT_CASES]
        self.panel.changed()
        self.refresh()

    def export_cases(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export load cases", "ft_cases.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.model.to_dict(), indent=1))

    def import_cases(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import load cases", "", "JSON (*.json)")
        if not path:
            return
        try:
            d = json.loads(Path(path).read_text())
            self.model.from_dict(d if isinstance(d, dict) else {"cases": d})
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, "F/T", f"Cannot read {path}:\n{e}")
            return
        self.panel.changed()
        self.refresh()

    def import_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, "Matrix from mini58_cal.py fit", "", "JSON (*.json)")
        if not path:
            return
        try:
            d = json.loads(Path(path).read_text())
            C = np.asarray(d["matrix"], float)                    # N per V at sensor output, sign-corrected
            sign = np.asarray(d.get("gage_sign", [1] * 6), float)
            ratio = np.asarray(d.get("front_end_ratio", [1] * 6), float)
            if C.shape != (6, 6):
                raise ValueError("matrix is not 6x6")
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, "F/T", f"Cannot read {path}:\n{e}")
            return
        Cmv = C * (sign * ratio)[None, :] / 1000.0
        self.model.imported = {"C": Cmv.tolist(), "note": Path(path).name}
        if "gage_sign" not in d:
            QMessageBox.information(self, "F/T", "No gage_sign in the file: assumed all +1. If your config "
                                                 "had signs, re-run fit with the current mini58_cal.py.")
        self.panel.changed()
        self.refresh()

    def drop_matrix(self):
        self.model.imported = None
        self.panel.changed()
        self.refresh()

    def apply_channels(self):
        try:
            ch = [int(x) for x in self.gage.text().replace(" ", "").split(",") if x != ""]
            if len(ch) != 6 or any(c < 0 or c > 7 for c in ch):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "F/T", "Gage channels: six numbers 0-7, e.g. 1,2,3,4,5,6")
            return
        m = self.model
        changed = ch != m.gage_ch or self.exc.currentData() != m.exc_ch
        m.gage_ch = ch
        m.exc_ch = self.exc.currentData()
        m.exc_ratio = self.exc_ratio.value()
        m.exc_nominal = self.exc_nom.value()
        m.V0 = None
        if changed and m.cases and QMessageBox.question(
                self, "F/T", "Channels changed: delete the existing load cases (they were captured with the "
                             "old setup)?") == QMessageBox.Yes:
            m.cases = []
        self.panel.changed()
        self.refresh()
