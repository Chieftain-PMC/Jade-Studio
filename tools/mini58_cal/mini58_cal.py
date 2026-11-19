#!/usr/bin/env python3
"""
mini58_cal.py - calibration tools for an ATI Mini58 (DAQ / analog version) read by Jade.

The Mini58 outputs six strain-gage voltages SG0..SG5. They are NOT Fx..Tz.
Forces and torques come from a 6x6 matrix:

    [Fx Fy Fz Tx Ty Tz]^T = C . (V - V0)

    V  = the six gage voltages (volts, at the sensor output)
    V0 = the same six voltages with no load (tare / bias)
    C  = 6x6 calibration matrix (ATI .cal file "UserAxis" rows, or fitted here)

Commands
    quicklook  LOG.csv                 stats per channel (offset, noise, headroom to ADC limit)
    rangecheck LOG.csv --mass 2 ...    tare / one weight / tare: projected max mass before the ADC limit
    segment    LOG.csv -o draft.csv    find stable plateaus in a continuous log, write a draft steps file
    axis       STEPS.csv --axis Fx     single-axis test: sensitivity, linearity, hysteresis, zero return
    fit        STEPS.csv [...]         full 6x6 least-squares fit of C (needs loads that excite all 6 axes)
    ati        FTxxxxx.cal STEPS.csv   apply the ATI matrix to your data and compare to the applied loads
    loads      STEPS.csv               print the load vector computed for every step (geometry check)

Every command reads the hardware description from a JSON config (default: jade_mini58.json
next to this script). See README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit("pandas is required:  pip install pandas numpy matplotlib")

AXES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
GAGES = ["SG0", "SG1", "SG2", "SG3", "SG4", "SG5"]

# ATI Mini58 metric calibrations (rated ranges, N and N.m), from the ATI Mini58 product page.
MINI58_CALS = {
    "SI-700-30":   {"Fx": 700,  "Fy": 700,  "Fz": 1700, "Tx": 30,  "Ty": 30,  "Tz": 30},
    "SI-1400-60":  {"Fx": 1400, "Fy": 1400, "Fz": 3400, "Tx": 60,  "Ty": 60,  "Tz": 60},
    "SI-2800-120": {"Fx": 2800, "Fy": 2800, "Fz": 6800, "Tx": 120, "Ty": 120, "Tz": 120},
}

# Colors (fixed order) for plots
C_UP = "#2a78d6"
C_DOWN = "#eb6834"
C_FIT = "#6b6a63"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


# --------------------------------------------------------------------------- config

DEFAULT_CONFIG = {
    "time_column": "time_s",
    "time_scale": 1.0,
    "sample_rate_hz": None,
    "gage_columns": ["CH0", "CH1", "CH2", "CH3", "CH4", "CH5"],
    "ref_column": None,
    "exc_column": None,
    "exc_pga_gain": 1,
    "exc_ratio": 3.0,
    "exc_nominal": 2.5,
    "csv_delimiter": ",",
    "units": "counts",
    "pga_gain": [1, 1, 1, 1, 1, 1],
    "ref_pga_gain": 1,
    "vref": 1.2,
    "front_end_ratio": [1, 1, 1, 1, 1, 1],
    "gage_sign": [1, 1, 1, 1, 1, 1],
    "adc_limit_fraction": 0.95,
    "g": 9.8066,
    "calibration": None,
    "trim_start_s": 3.0,
    "trim_end_s": 1.0,
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jade_mini58.json")
        path = here if os.path.exists(here) else None
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        user = {k: v for k, v in user.items() if not k.startswith("_")}
        cfg.update(user)
    for key in ("pga_gain", "front_end_ratio", "gage_sign"):
        v = cfg[key]
        if not isinstance(v, list):
            cfg[key] = [v] * 6
        if len(cfg[key]) != 6:
            sys.exit(f"config: '{key}' must have 6 values")
    if len(cfg["gage_columns"]) != 6:
        sys.exit("config: 'gage_columns' must list 6 columns (SG0..SG5 in order)")
    return cfg


def rated_ranges(cfg: dict) -> dict | None:
    cal = cfg.get("calibration")
    if cal is None:
        return None
    if isinstance(cal, dict):
        return cal
    if cal in MINI58_CALS:
        return MINI58_CALS[cal]
    sys.exit(f"config: unknown calibration '{cal}' (use one of {list(MINI58_CALS)} or a dict)")


# --------------------------------------------------------------------------- data loading

_LOG_CACHE: dict = {}


def adc_to_volts(raw: np.ndarray, units: str, gain: float, vref: float) -> np.ndarray:
    """Convert one ADC column to volts at the ADC input pins."""
    raw = raw.astype(float)
    if units == "counts":           # signed 24-bit two's complement already sign-extended
        return raw / 8388608.0 * vref / gain
    if units == "counts_u24":       # raw 24-bit word as unsigned 0..16777215
        raw = np.where(raw >= 8388608, raw - 16777216, raw)
        return raw / 8388608.0 * vref / gain
    if units == "V":
        return raw
    if units == "mV":
        return raw / 1000.0
    if units == "uV":
        return raw / 1e6
    sys.exit(f"unknown units '{units}' (counts, counts_u24, V, mV, uV)")


@dataclass
class Log:
    path: str
    t: np.ndarray                 # seconds
    v_adc: np.ndarray             # (n, 6) volts at ADC pins (differential)
    v_gage: np.ndarray            # (n, 6) volts at the sensor output (after front-end ratio, + ref)
    v_ref: np.ndarray | None = None
    gains: list | None = None


def read_log(path: str, cfg: dict) -> Log:
    key = os.path.abspath(path)
    if key in _LOG_CACHE:
        return _LOG_CACHE[key]
    if path.lower().endswith(".bin"):
        df = read_jade_bin(path, float(cfg["vref"]))
        print(f"{os.path.basename(path)}: Jade .BIN, {len(df)} samples at {df.attrs['rate']:.0f} S/s, "
              f"PGA gains from the file {df.attrs['gains']}")
        cfg = dict(cfg, units="V", time_column="time_s", time_scale=1.0,
                   pga_gain=[df.attrs["gains"].get(c, 1) for c in cfg["gage_columns"]])
    else:
        df = pd.read_csv(path, sep=cfg["csv_delimiter"], comment="#", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in cfg["gage_columns"] if c not in df.columns]
    if missing:
        sys.exit(f"{path}: columns {missing} not found. Columns in file: {list(df.columns)}\n"
                 f"Edit 'gage_columns' in the config.")
    tcol = cfg.get("time_column")
    if tcol and tcol in df.columns:
        t = df[tcol].to_numpy(float) * float(cfg.get("time_scale", 1.0))
    elif cfg.get("sample_rate_hz"):
        t = np.arange(len(df)) / float(cfg["sample_rate_hz"])
    else:
        sys.exit(f"{path}: time column '{tcol}' not found and no sample_rate_hz in config")
    t = t - t[0]

    v_adc = np.column_stack([
        adc_to_volts(df[c].to_numpy(), cfg["units"], float(cfg["pga_gain"][i]), float(cfg["vref"]))
        for i, c in enumerate(cfg["gage_columns"])
    ])
    # gage_sign = -1 where the gage wire is on AINxN and the reference on AINxP
    v_gage = v_adc * (np.asarray(cfg["front_end_ratio"], float) * np.asarray(cfg["gage_sign"], float))[None, :]
    v_ref = None
    rc = cfg.get("ref_column")
    if rc:
        if rc not in df.columns:
            sys.exit(f"{path}: ref_column '{rc}' not found")
        v_ref = adc_to_volts(df[rc].to_numpy(), cfg["units"], float(cfg["ref_pga_gain"]), float(cfg["vref"]))
        # each gage channel measures (SG - Vref); adding the measured Vref removes its noise/drift
        v_gage = v_gage + v_ref[:, None]
    ec = cfg.get("exc_column")
    if ec:
        if ec not in df.columns:
            sys.exit(f"{path}: exc_column '{ec}' not found")
        v_exc = adc_to_volts(df[ec].to_numpy(), cfg["units"], float(cfg["exc_pga_gain"]), float(cfg["vref"])) \
            * float(cfg["exc_ratio"])
        if np.any(v_exc <= 0.1 * float(cfg["exc_nominal"])):
            sys.exit(f"{path}: excitation channel '{ec}' reads near zero - check exc_column / exc_ratio")
        # ratiometric: half-bridge output scales with excitation, the ADC reference does not
        v_gage = v_gage * (float(cfg["exc_nominal"]) / v_exc)[:, None]
    log = Log(path, t, v_adc, v_gage, v_ref, gains=[float(g) for g in cfg["pga_gain"]])
    _LOG_CACHE[key] = log
    return log


# --------------------------------------------------------------------------- Jade SD .BIN

def _cobs_decode(b: bytes) -> bytes:
    out = bytearray(); i = 0
    while i < len(b):
        c = b[i]
        out += b[i + 1:i + c]
        i += c
        if c < 0xFF and i < len(b):
            out.append(0)
    return bytes(out)


def _crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def read_jade_bin(path: str, vref: float) -> "pd.DataFrame":
    """Jade SD log: COBS frames separated by 0x00, CRC16-CCITT at the end of each frame.
    0x80 = ADC config (channel mask at byte 8, PGA gain per channel at bytes 9..16); it can change mid-file,
    and every following sample uses the gains in force when it was taken.
    0x10 = samples: seq u16, t_ms u32, t_us u32, mask u8, n u8, n x (flag u8 + 3-byte LE signed per channel).
    Returns time_s and CH0..CH7 in volts at the ADC pins (NaN where a channel was off)."""
    raw = open(path, "rb").read()
    gains = [1] * 8
    frames, changes, bad = [], [], 0
    for f in raw.split(b"\x00"):
        if not f:
            continue
        fr = _cobs_decode(f)
        if len(fr) < 4 or _crc16_ccitt(fr[:-2]) != int.from_bytes(fr[-2:], "little"):
            bad += 1
            continue
        if fr[0] == 0x80 and len(fr) >= 17:
            g = [x if x in (1, 2, 4, 8, 16, 32, 64, 128) else 1 for x in fr[9:17]]
            if g != gains or not changes:
                changes.append((len(frames), g, fr[1:9].hex(" ")))
            gains = g
        if fr[0] != 0x10:
            continue
        t_us = int.from_bytes(fr[8:12], "little")
        mask, n = fr[12], fr[13]
        chans = [i for i in range(8) if mask >> i & 1]
        sz = 1 + 3 * len(chans)
        vals = np.full((n, 8), np.nan)
        for k in range(n):
            smp = fr[14 + k * sz:14 + (k + 1) * sz]
            for j, ch in enumerate(chans):
                vals[k, ch] = int.from_bytes(smp[1 + 3 * j:4 + 3 * j], "little", signed=True) / 8388608.0 * vref / gains[ch]
        frames.append((t_us, n, vals, tuple(gains)))
    if not frames:
        sys.exit(f"{path}: no sample frames found")
    # per-frame sample spacing from the next frame's start (handles data-rate changes and u32 wrap)
    ts, vs, gs = [], [], []
    prev_dt = 4000.0
    for i, (t0, n, vals, g) in enumerate(frames):
        if i + 1 < len(frames):
            dt_next = (frames[i + 1][0] - t0) % (1 << 32)
            dt = dt_next / n if 0 < dt_next / n < 1e6 else prev_dt
        else:
            dt = prev_dt
        prev_dt = dt
        ts.append(t0 + dt * np.arange(n))
        vs.append(vals)
        gs += [g] * n
    t = np.concatenate(ts)
    t = np.unwrap(t / (1 << 32) * 2 * np.pi) / (2 * np.pi) * (1 << 32) / 1e6
    V = np.vstack(vs)
    df = pd.DataFrame({"time_s": t - t[0]})
    used = [c for c in range(8) if not np.all(np.isnan(V[:, c]))]
    for c in used:
        df[f"CH{c}"] = V[:, c]
    if bad:
        print(f"note: {bad} frame(s) with a bad CRC skipped in {os.path.basename(path)}")
    df.attrs["gains"] = {f"CH{c}": gains[c] for c in used}
    df.attrs["rate"] = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else float("nan")
    df.attrs["changes"] = [(float(t[sum(fr[1] for fr in frames[:i])] - t[0]) if i < len(frames) else None, g, h)
                           for i, g, h in changes]
    if len(changes) > 1:
        print(f"note: ADC settings changed {len(changes) - 1} time(s) in {os.path.basename(path)}; "
              "each sample is scaled with the gain in force when it was taken")
    return df


# --------------------------------------------------------------------------- steps / loads

DIRS = {
    "+x": (1, 0, 0), "-x": (-1, 0, 0), "x": (1, 0, 0),
    "+y": (0, 1, 0), "-y": (0, -1, 0), "y": (0, 1, 0),
    "+z": (0, 0, 1), "-z": (0, 0, -1), "z": (0, 0, 1),
}


def parse_dir(s) -> np.ndarray:
    s = str(s).strip().lower().replace(" ", "")
    if s in DIRS:
        return np.array(DIRS[s], float)
    parts = s.replace(",", ";").split(";")
    if len(parts) == 3:
        v = np.array([float(p) for p in parts])
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError("zero direction vector")
        return v / n
    raise ValueError(f"cannot parse direction '{s}' (use +x, -y, ... or 'vx;vy;vz')")


def load_vector(mass_kg: float, direction: np.ndarray, r_mm: np.ndarray, g: float) -> np.ndarray:
    """6-vector [Fx Fy Fz Tx Ty Tz] (N, N.m) of a mass hung at point r (sensor frame)."""
    F = mass_kg * g * direction
    T = np.cross(r_mm / 1000.0, F)
    return np.concatenate([F, T])


@dataclass
class Step:
    idx: int
    step: str
    series: str
    file: str | None
    t_start: float | None
    t_end: float | None
    mass: float
    direction: np.ndarray
    r_mm: np.ndarray
    note: str = ""
    # filled by process_steps
    L: np.ndarray = field(default_factory=lambda: np.zeros(6))
    V: np.ndarray = field(default_factory=lambda: np.zeros(6))
    V_std: np.ndarray = field(default_factory=lambda: np.zeros(6))
    V_adc_peak: np.ndarray = field(default_factory=lambda: np.zeros(6))
    n: int = 0
    t_mid: float = 0.0
    dV: np.ndarray = field(default_factory=lambda: np.zeros(6))
    saturated: bool = False
    branch: str = ""
    cycle: int = 0


def _cell(row, key):
    v = row.get(key, None)
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s != "" else None


def read_steps(path: str, default_log: str | None) -> list[Step]:
    df = pd.read_csv(path, comment="#", skipinitialspace=True, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    need = ["mass_kg", "dir"]
    for c in need:
        if c not in df.columns:
            sys.exit(f"{path}: missing column '{c}'")
    base = os.path.dirname(os.path.abspath(path))
    steps = []
    for i, row in df.iterrows():
        row = row.to_dict()
        if all(str(v).strip() == "" for v in row.values()):
            continue
        f = _cell(row, "file") or default_log
        if f is None:
            sys.exit(f"{path}: step {i+1} has no 'file' and no --log was given")
        if not os.path.isabs(f):
            f_rel = os.path.join(base, f)
            f = f_rel if os.path.exists(f_rel) else f
        ts, te = _cell(row, "t_start"), _cell(row, "t_end")
        steps.append(Step(
            idx=len(steps),
            step=_cell(row, "step") or str(len(steps) + 1),
            series=_cell(row, "series") or "A",
            file=f,
            t_start=float(ts) if ts is not None else None,
            t_end=float(te) if te is not None else None,
            mass=float(_cell(row, "mass_kg") or 0.0),
            direction=parse_dir(_cell(row, "dir") or "-z"),
            r_mm=np.array([float(_cell(row, k) or 0.0) for k in ("x_mm", "y_mm", "z_mm")]),
            note=_cell(row, "note") or "",
        ))
    if not steps:
        sys.exit(f"{path}: no steps")
    return steps


def window_of(step: Step, log: Log, cfg: dict) -> np.ndarray:
    if step.t_start is not None and step.t_end is not None:
        m = (log.t >= step.t_start) & (log.t <= step.t_end)
    else:  # one file per step: drop the settling at the start and the handling at the end
        t0 = log.t[0] + float(cfg["trim_start_s"])
        t1 = log.t[-1] - float(cfg["trim_end_s"])
        m = (log.t >= t0) & (log.t <= t1)
    if m.sum() < 10:
        sys.exit(f"step {step.step}: window has {int(m.sum())} samples - check t_start/t_end or the file")
    return m


def adc_limits(log: Log, cfg: dict, fraction: float = 1.0) -> np.ndarray:
    return np.array([float(cfg["vref"]) / g for g in log.gains]) * fraction


def process_steps(steps: list[Step], cfg: dict, tare: str = "first") -> list[Step]:
    """Average each window, compute applied load, subtract the tare of each series."""
    g = float(cfg["g"])
    for s in steps:
        log = read_log(s.file, cfg)
        lim = adc_limits(log, cfg, float(cfg["adc_limit_fraction"]))
        m = window_of(s, log, cfg)
        s.V = log.v_gage[m].mean(axis=0)
        s.V_std = log.v_gage[m].std(axis=0)
        s.V_adc_peak = np.abs(log.v_adc[m]).max(axis=0)
        s.saturated = bool(np.any(s.V_adc_peak >= lim))
        s.n = int(m.sum())
        s.t_mid = float(log.t[m].mean())
        s.L = load_vector(s.mass, s.direction, s.r_mm, g)

    # tare per series
    by_series: dict[str, list[Step]] = {}
    for s in steps:
        by_series.setdefault(s.series, []).append(s)
    for name, ss in by_series.items():
        zeros = [s for s in ss if s.mass == 0]
        if not zeros:
            sys.exit(f"series '{name}': needs at least one step with mass_kg = 0 (the tare)")
        if tare == "first" or len(zeros) == 1:
            V0 = zeros[0].V
            for s in ss:
                s.dV = s.V - V0
        elif tare == "drift":
            # linear interpolation of the zero between successive zero steps (by step order)
            zi = np.array([ss.index(z) for z in zeros], float)
            zV = np.array([z.V for z in zeros])
            for k, s in enumerate(ss):
                V0 = np.array([np.interp(k, zi, zV[:, j]) for j in range(6)])
                s.dV = s.V - V0
        else:
            sys.exit(f"unknown tare mode '{tare}'")

        # branch / cycle labelling for hysteresis
        prev, cyc, branch, seen_load = 0.0, 0, "zero", False
        for s in ss:
            if s.mass > prev:
                branch = "up"
            elif s.mass < prev:
                branch = "down"
            if s.mass == 0:
                if seen_load:
                    s.branch = "zero-return"
                    cyc += 1
                    seen_load = False
                else:
                    s.branch = "zero"
                s.cycle = cyc
            else:
                seen_load = True
                s.branch = branch
                s.cycle = cyc
            prev = s.mass
    return steps


# --------------------------------------------------------------------------- quicklook

def cmd_quicklook(a, cfg):
    log = read_log(a.log, cfg)
    m = np.ones(len(log.t), bool)
    if a.t0 is not None:
        m &= log.t >= a.t0
    if a.t1 is not None:
        m &= log.t <= a.t1
    tm = log.t[m]
    fs = (len(tm) - 1) / (tm[-1] - tm[0]) if len(tm) > 1 and tm[-1] > tm[0] else float("nan")
    lim = adc_limits(log, cfg)
    print(f"{log.path}: {m.sum()} samples, {log.t[m][0]:.2f}..{log.t[m][-1]:.2f} s, ~{fs:.1f} S/s\n")
    print(f"{'ch':<5}{'column':<10}{'V_gage mean':>13}{'noise rms':>12}{'p-p':>11}{'ADC |peak|':>12}{'headroom':>10}")
    for i in range(6):
        vg = log.v_gage[m, i]
        va = log.v_adc[m, i]
        pk = np.abs(va).max()
        print(f"{GAGES[i]:<5}{cfg['gage_columns'][i]:<10}{vg.mean():>12.5f}V{vg.std()*1e6:>10.1f}uV"
              f"{(vg.max()-vg.min())*1e3:>9.3f}mV{pk:>11.4f}V{100*(1-pk/lim[i]):>9.1f}%")
    if log.v_ref is not None:
        r = log.v_ref[m]
        print(f"\nref  {cfg['ref_column']:<10}{r.mean():>12.5f}V{r.std()*1e6:>10.1f}uV")
    off = np.abs(log.v_gage[m].mean(axis=0))
    if np.any(off > 1.0):
        print("\nWARNING: unloaded gage offset above 1 V on", [GAGES[i] for i in np.where(off > 1.0)[0]],
              "- ATI says healthy unloaded gages normally sit within +/-1 V. Check wiring/power, or the sensor.")
    print("\nheadroom = how much of the ADC differential range (Vref/gain) is still free on each channel.")


# --------------------------------------------------------------------------- segment

def find_plateaus(log, window=1.0, k=4.0, min_hold=5.0, merge_gap=2.0):
    """Stable stretches of a log: list of (t0, t1)."""
    t, V = log.t, log.v_gage
    fs = (len(t) - 1) / (t[-1] - t[0])
    w = max(5, int(round(window * fs)))

    def mov(x, n):
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[n:] - c[:-n]) / n
    stds = []
    for j in range(6):
        x = V[:, j] - V[:, j].mean()
        mu = mov(x, w)
        mu2 = mov(x * x, w)
        stds.append(np.sqrt(np.maximum(mu2 - mu * mu, 0)))
    S = np.column_stack(stds)
    floor = np.median(S, axis=0) + 1e-12
    score = (S / floor).max(axis=1)
    steady = score < k
    tc = t[w // 2: w // 2 + len(score)]
    runs, start = [], None
    for i, ok in enumerate(steady):
        if ok and start is None:
            start = i
        if (not ok or i == len(steady) - 1) and start is not None:
            if tc[i] - tc[start] >= min_hold:
                runs.append((tc[start], tc[i]))
            start = None
    merged = []
    for r in runs:
        if merged:
            p = merged[-1]
            mp = (t >= p[0]) & (t <= p[1])
            mr = (t >= r[0]) & (t <= r[1])
            d = np.abs(V[mp].mean(0) - V[mr].mean(0)) / floor   # in noise sigmas
            if r[0] - p[1] < merge_gap and np.all(d < 10):
                merged[-1] = (p[0], r[1])
                continue
        merged.append(r)
    return merged


def cmd_segment(a, cfg):
    log = read_log(a.log, cfg)
    t, V = log.t, log.v_gage
    merged = find_plateaus(log, a.window, a.k, a.min_hold, a.merge_gap)
    rows = []
    for k, (t0, t1) in enumerate(merged):
        t0 += a.edge
        t1 -= a.edge
        if t1 - t0 < 1.0:
            continue
        mm = (t >= t0) & (t <= t1)
        mv = V[mm].mean(0)
        rows.append({"step": len(rows) + 1, "series": "A", "file": os.path.basename(a.log),
                     "t_start": round(t0, 2), "t_end": round(t1, 2), "mass_kg": "",
                     "dir": a.dir, "x_mm": 0, "y_mm": 0, "z_mm": 0,
                     "note": "V=" + " ".join(f"{x:+.4f}" for x in mv)})
    if a.plan:
        plan = pd.read_csv(a.plan, comment="#", skipinitialspace=True, dtype=str).fillna("")
        plan.columns = [c.strip() for c in plan.columns]
        plan = plan[~(plan == "").all(axis=1)].reset_index(drop=True)
        if len(plan) != len(rows):
            print(f"WARNING: found {len(rows)} plateaus but the plan has {len(plan)} steps. Masses were NOT"
                  f" filled in - look at the plot, fix the rows by hand (or tune --k / --min-hold).")
        else:
            for r, (_, p) in zip(rows, plan.iterrows()):
                for c in ("step", "series", "mass_kg", "dir", "x_mm", "y_mm", "z_mm"):
                    if c in plan.columns and str(p[c]).strip() != "":
                        r[c] = p[c]
                if "note" in plan.columns and str(p["note"]).strip():
                    r["note"] = str(p["note"]).strip() + " | " + r["note"]
            print(f"plan {a.plan}: {len(plan)} steps matched to the plateaus in order - check the plot anyway.")
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"{len(rows)} plateaus written to {a.out}. Check mass_kg, dir and the lever arm, delete bad rows,"
          f" then run 'axis' or 'fit'.")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for j in range(6):
            ax.plot(t, V[:, j], lw=1, color=SERIES_COLORS[j], label=GAGES[j])
        for r in rows:
            ax.axvspan(r["t_start"], r["t_end"], color="#c3c2b7", alpha=0.35, lw=0)
            ax.text((r["t_start"] + r["t_end"]) / 2, ax.get_ylim()[1], str(r["step"]),
                    ha="center", va="bottom", fontsize=7, rotation=90, color="#3d3c36")
        _style(ax, "time (s)", "gage voltage (V)")
        ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.15))
        fig.tight_layout()
        p = os.path.splitext(a.out)[0] + "_segments.png"
        fig.savefig(p, dpi=130)
        print("plot:", p)


# --------------------------------------------------------------------------- helpers

def _style(ax, xl, yl):
    ax.set_xlabel(xl, fontsize=9, color="#3d3c36")
    ax.set_ylabel(yl, fontsize=9, color="#3d3c36")
    ax.grid(True, color="#e4e3dc", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9b9a92")
    ax.tick_params(labelsize=8, colors="#3d3c36")


def steps_table(steps: list[Step]) -> pd.DataFrame:
    rows = []
    for s in steps:
        r = {"step": s.step, "series": s.series, "cycle": s.cycle, "branch": s.branch,
             "mass_kg": s.mass, "n": s.n, "t_mid_s": round(s.t_mid, 2), "saturated": s.saturated}
        for k, ax in enumerate(AXES):
            r[f"L_{ax}"] = s.L[k]
        for k, gname in enumerate(GAGES):
            r[f"V_{gname}"] = s.V[k]
        for k, gname in enumerate(GAGES):
            r[f"dV_{gname}"] = s.dV[k]
        for k, gname in enumerate(GAGES):
            r[f"std_{gname}_uV"] = s.V_std[k] * 1e6
        rows.append(r)
    return pd.DataFrame(rows)


def warn_saturation(steps):
    bad = [s.step for s in steps if s.saturated]
    if bad:
        print(f"WARNING: ADC near/at its limit in steps {bad}. Those points are excluded - reduce load"
              f" or attenuate the inputs.")
    return [s for s in steps if not s.saturated]


# --------------------------------------------------------------------------- axis analysis

def cmd_axis(a, cfg):
    steps = []
    for p in a.steps:
        steps += read_steps(p, a.log)
    steps = warn_saturation(process_steps(steps, cfg, tare=a.tare))
    k = AXES.index(a.axis)
    unit = "N" if k < 3 else "N.m"
    os.makedirs(a.out, exist_ok=True)

    F = np.array([s.L[k] for s in steps])
    D = np.array([s.dV for s in steps])                      # (n, 6)
    loaded = np.array([s.mass != 0 for s in steps])
    if np.ptp(F) == 0:
        sys.exit(f"no {a.axis} load in these steps - check 'dir' and the lever arm columns")

    # other components that ride along with this load case (lever-arm torques etc.)
    L = np.array([s.L for s in steps])
    ride = {}
    for j in range(6):
        if j == k or np.allclose(L[:, j], 0, atol=1e-9):
            continue
        c = np.polyfit(F, L[:, j], 1)[0]
        ride[AXES[j]] = c

    # per-gage straight-line fit dV = s*F + b
    sens, offs, r2, nl = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    span = np.ptp(D, axis=0)
    for j in range(6):
        p = np.polyfit(F, D[:, j], 1)
        sens[j], offs[j] = p
        res = D[:, j] - np.polyval(p, F)
        ss_tot = np.sum((D[:, j] - D[:, j].mean()) ** 2)
        r2[j] = 1 - np.sum(res ** 2) / ss_tot if ss_tot > 0 else float("nan")
        nl[j] = np.max(np.abs(res)) / span[j] * 100 if span[j] > 0 else float("nan")
    strong = span >= 0.2 * span.max()

    # scalar estimator for this load case: F_hat = s.dV / (s.s)   (least squares projection)
    Fhat = D @ sens / (sens @ sens)
    p_est = np.polyfit(F, Fhat, 1)
    err = Fhat - F
    test_span = np.ptp(F)

    # hysteresis: same series, same cycle, same mass, up vs down
    hyst_rows = []
    key = {}
    for s, fh in zip(steps, Fhat):
        if s.branch in ("up", "down"):
            key.setdefault((s.series, s.cycle, round(s.mass, 6)), {}).setdefault(s.branch, []).append((s, fh))
    for (ser, cyc, mass), d in sorted(key.items()):
        if "up" in d and "down" in d:
            su, fu = d["up"][-1]
            sd, fd = d["down"][0]
            g_h = (sd.dV - su.dV)
            hyst_rows.append({"series": ser, "cycle": cyc, "mass_kg": mass, "load": su.L[k],
                              "est_up": fu, "est_down": fd, "hyst": fd - fu,
                              **{f"dV_{GAGES[j]}_uV": g_h[j] * 1e6 for j in range(6)}})
    hyst = pd.DataFrame(hyst_rows)

    # zero return per cycle
    zr_rows = []
    for s, fh in zip(steps, Fhat):
        if s.branch == "zero-return":
            zr_rows.append({"series": s.series, "after_cycle": s.cycle - 1, "step": s.step,
                            "est": fh, **{f"dV_{GAGES[j]}_uV": s.dV[j] * 1e6 for j in range(6)}})
    zr = pd.DataFrame(zr_rows)

    # repeatability: same series / mass / branch across cycles
    rep = []
    grp = {}
    for s, fh in zip(steps, Fhat):
        if s.branch in ("up", "down"):
            grp.setdefault((s.series, round(s.mass, 6), s.branch), []).append(fh)
    for (ser, mass, br), v in grp.items():
        if len(v) >= 2:
            rep.append(np.ptp(v))
    repeat = max(rep) if rep else float("nan")

    # projected load at which a channel hits the ADC limit
    lim = adc_limits(read_log(steps[0].file, cfg), cfg, float(cfg["adc_limit_fraction"]))
    ratio = np.asarray(cfg["front_end_ratio"], float)
    V0_adc = np.array([steps[0].V[j] / ratio[j] for j in range(6)])
    if cfg.get("ref_column"):
        V0_adc = V0_adc * np.nan  # with a measured reference the absolute ADC level is not V/ratio
    sens_adc = sens / ratio
    max_pos, max_neg = [], []
    for j in range(6):
        if abs(sens_adc[j]) < 1e-15 or np.isnan(V0_adc[j]):
            continue
        for sign, lst in ((1, max_pos), (-1, max_neg)):
            # find F such that |V0 + s F| = lim, with F of given sign
            cands = [(lim[j] - V0_adc[j]) / sens_adc[j], (-lim[j] - V0_adc[j]) / sens_adc[j]]
            cands = [c for c in cands if c * sign > 0]
            if cands:
                lst.append(min(cands, key=abs))
    fmax_pos = min(max_pos, key=abs) if max_pos else float("nan")
    fmax_neg = min(max_neg, key=abs) if max_neg else float("nan")

    rated = rated_ranges(cfg)
    fs_txt = ""
    if rated:
        fs = rated[a.axis]
        fs_txt = f" (rated {a.axis} = {fs} {unit})"

    # ---------- report
    lines = []
    P = lines.append
    P(f"# {a.axis} single-axis test")
    P("")
    P(f"Steps: {len(steps)} ({int(loaded.sum())} loaded) from {', '.join(a.steps)}. Tare mode: {a.tare}.")
    P(f"Test range: {F.min():+.2f} to {F.max():+.2f} {unit}{fs_txt}.")
    if ride:
        P("Components that ride along with this load (from the lever arm geometry): " +
          ", ".join(f"{ax} = {c*1000:+.2f} mN.m per N" if ax.startswith("T") else f"{ax} = {c:+.4f} N/N"
                    for ax, c in ride.items()) +
          ". The gage slopes below are for this combined load case, not pure " + a.axis + ".")
    P("")
    P("## Gage response")
    P("")
    P("| gage | slope (uV/" + unit + ") | R^2 | span (mV) | nonlinearity (% of span) |")
    P("|---|---:|---:|---:|---:|")
    for j in range(6):
        nltxt = f"{nl[j]:.3f}" if strong[j] else f"({nl[j]:.2f}, low response)"
        P(f"| {GAGES[j]} | {sens[j]*1e6:+.2f} | {r2[j]:.6f} | {span[j]*1e3:.3f} | {nltxt} |")
    P("")
    P(f"## {a.axis} estimated from the 6 gages (projection on the slope vector)")
    P("")
    P(f"Estimator slope vs applied: {p_est[0]:.5f} (should be 1.00000), offset {p_est[1]:+.3f} {unit}.")
    P(f"Max |error| over all points: {np.max(np.abs(err)):.3f} {unit} = "
      f"{np.max(np.abs(err))/test_span*100:.3f} % of test span"
      + (f" = {np.max(np.abs(err))/rated[a.axis]*100:.3f} % FS" if rated else "") + ".")
    if len(hyst):
        hmax = hyst["hyst"].abs().max()
        P(f"Max hysteresis (down minus up, same load, same cycle): {hmax:.3f} {unit} = "
          f"{hmax/test_span*100:.3f} % of test span" + (f" = {hmax/rated[a.axis]*100:.3f} % FS" if rated else "") + ".")
    else:
        P("No hysteresis pairs found (need the same mass on the way up and on the way down in one cycle).")
    if len(zr):
        P(f"Max zero return error: {zr['est'].abs().max():.3f} {unit}.")
    if not math.isnan(repeat):
        P(f"Repeatability (worst spread of the same point across cycles): {repeat:.3f} {unit}.")
    P(f"Noise during holds (worst gage, rms): {max(s.V_std.max() for s in steps)*1e6:.1f} uV.")
    if not (math.isnan(fmax_pos) and math.isnan(fmax_neg)):
        P(f"Projected load where the first ADC channel reaches {cfg['adc_limit_fraction']*100:.0f} % of its range: "
          f"{fmax_pos:+.1f} {unit} / {fmax_neg:+.1f} {unit}"
          + (f"  (= {fmax_pos/float(cfg['g']):+.2f} kg / {fmax_neg/float(cfg['g']):+.2f} kg hung in this setup)" if k < 3 else "")
          + ".")
    P("")
    if len(hyst):
        P("## Hysteresis points")
        P("")
        P("| series | cycle | mass (kg) | load | up | down | down-up |")
        P("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in hyst.iterrows():
            P(f"| {r.series} | {int(r.cycle)} | {r.mass_kg:g} | {r.load:+.3f} | {r.est_up:+.3f} | {r.est_down:+.3f} | {r.hyst:+.3f} |")
        P("")
    P("## Interim single-axis formula")
    P("")
    P(f"Valid only for this load case (same lever arm, no other loads). Use the full 6x6 fit or the ATI matrix for real use.")
    P("")
    coef = sens / (sens @ sens)
    P(f"    {a.axis} = " + " ".join(f"{c:+.6g}*d{GAGES[j]}" for j, c in enumerate(coef)) + f"   [{unit}, dSGi in V]")
    P("")
    report = "\n".join(lines)
    with open(os.path.join(a.out, f"{a.axis}_report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    tab = steps_table(steps)
    tab[f"est_{a.axis}"] = Fhat
    tab.to_csv(os.path.join(a.out, f"{a.axis}_steps.csv"), index=False)
    if len(hyst):
        hyst.to_csv(os.path.join(a.out, f"{a.axis}_hysteresis.csv"), index=False)
    res = {"axis": a.axis, "unit": unit, "slope_V_per_unit": dict(zip(GAGES, sens.tolist())),
           "r2": dict(zip(GAGES, r2.tolist())), "nonlinearity_pct_span": dict(zip(GAGES, nl.tolist())),
           "estimator_coefficients": dict(zip(GAGES, coef.tolist())),
           "max_error": float(np.max(np.abs(err))),
           "max_hysteresis": float(hyst["hyst"].abs().max()) if len(hyst) else None,
           "test_span": float(test_span), "ride_along": ride}
    with open(os.path.join(a.out, f"{a.axis}_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(report)

    if not a.no_plot:
        _plot_axis(a, steps, F, D, sens, offs, Fhat, unit)
        print(f"\nplots and tables in {a.out}/")


def _plot_axis(a, steps, F, D, sens, offs, Fhat, unit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    br = np.array([s.branch for s in steps])
    up = np.isin(br, ["up", "zero"])
    dn = np.isin(br, ["down", "zero-return"])

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    xx = np.linspace(F.min(), F.max(), 50)
    for j, ax in enumerate(axs.flat):
        ax.plot(xx, (sens[j] * xx + offs[j]) * 1e3, color=C_FIT, lw=1, zorder=1)
        ax.scatter(F[up], D[up, j] * 1e3, marker="^", s=36, color=C_UP, edgecolor="white", lw=1, zorder=3, label="loading")
        ax.scatter(F[dn], D[dn, j] * 1e3, marker="v", s=36, color=C_DOWN, edgecolor="white", lw=1, zorder=3, label="unloading")
        ax.set_title(f"{GAGES[j]}   {sens[j]*1e6:+.1f} uV/{unit}", fontsize=9, loc="left", color="#1f1e1b")
        _style(ax, f"applied {a.axis} ({unit})" if j >= 3 else "", "dV (mV)" if j % 3 == 0 else "")
    axs.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Gage response to {a.axis} load case", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{a.axis}_gages.png"), dpi=130)
    plt.close(fig)

    # error / hysteresis loop
    fig, ax = plt.subplots(figsize=(8, 4.5))
    err = Fhat - F
    for ser in sorted(set(s.series for s in steps)):
        for cyc in sorted(set(s.cycle for s in steps if s.series == ser)):
            idx = [i for i, s in enumerate(steps) if s.series == ser and s.cycle == cyc]
            ax.plot(F[idx], err[idx], color="#c3c2b7", lw=1, zorder=1)
    ax.scatter(F[up], err[up], marker="^", s=40, color=C_UP, edgecolor="white", lw=1, zorder=3, label="loading")
    ax.scatter(F[dn], err[dn], marker="v", s=40, color=C_DOWN, edgecolor="white", lw=1, zorder=3, label="unloading")
    ax.axhline(0, color="#9b9a92", lw=0.8)
    _style(ax, f"applied {a.axis} ({unit})", f"estimated - applied ({unit})")
    ax.set_title(f"{a.axis} error and hysteresis loop (straight-line fit through all points)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{a.axis}_hysteresis.png"), dpi=130)
    plt.close(fig)

    # raw time series with windows, one figure per file
    files = []
    for s in steps:
        if s.file not in files:
            files.append(s.file)
    if len(files) <= 3:
        for fpath in files:
            log = _LOG_CACHE[os.path.abspath(fpath)]
            fig, ax = plt.subplots(figsize=(12, 4))
            dec = max(1, len(log.t) // 20000)
            for j in range(6):
                ax.plot(log.t[::dec], log.v_gage[::dec, j], lw=0.8, color=SERIES_COLORS[j], label=GAGES[j])
            for s in steps:
                if s.file == fpath and s.t_start is not None:
                    ax.axvspan(s.t_start, s.t_end, color="#c3c2b7", alpha=0.35, lw=0)
            _style(ax, "time (s)", "gage voltage (V)")
            ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))
            ax.set_title(os.path.basename(fpath) + " - averaged windows shaded", fontsize=10, loc="left")
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f"{a.axis}_timeseries_{os.path.splitext(os.path.basename(fpath))[0]}.png"), dpi=110)
            plt.close(fig)


# --------------------------------------------------------------------------- full fit

def cmd_fit(a, cfg):
    steps = []
    for p in a.steps:
        steps += read_steps(p, a.log)
    use = warn_saturation(process_steps(steps, cfg, tare=a.tare))
    D = np.array([s.dV for s in use])
    L = np.array([s.L for s in use])
    n = len(use)

    # how well do the load cases cover the 6 axes? (scale forces and torques comparably)
    scale = np.array([1, 1, 1, 1 / a.arm, 1 / a.arm, 1 / a.arm])
    sv = np.linalg.svd(L * scale[None, :], compute_uv=True)
    U, Sg, Vt = sv
    rel = Sg / Sg[0] if Sg[0] > 0 else Sg
    print(f"{n} load points. Load-coverage singular values (relative): " + " ".join(f"{x:.3g}" for x in rel))
    if n < 6 or rel[-1] < 0.02:
        print("NOT ENOUGH INDEPENDENT LOADS for a 6x6 fit. Poorly excited load combinations:")
        for kk in range(len(rel)):
            if rel[kk] < 0.02:
                weak = Vt[kk] / scale
                weak = weak / np.max(np.abs(weak))
                print("   " + "  ".join(f"{AXES[j]}:{weak[j]:+.2f}" for j in range(6)))
        print("Add load cases that move these combinations (other orientation, other lever arm).")
        if not a.force:
            sys.exit(1)

    # C^T = argmin || D C^T - L ||   (forces and torques fitted column by column)
    Ct, *_ = np.linalg.lstsq(D, L, rcond=None)
    C = Ct.T
    Lhat = D @ Ct
    E = Lhat - L
    cond = np.linalg.cond(C)

    rated = rated_ranges(cfg)
    lines = []
    P = lines.append
    P("# Mini58 6x6 calibration fit")
    P("")
    P(f"{n} load points from {', '.join(a.steps)}; tare mode {a.tare}; condition number of C = {cond:.1f}.")
    P("")
    P("    [Fx Fy Fz Tx Ty Tz]^T = C . (V - V0)      V in volts at the sensor output, F in N, T in N.m")
    P("")
    P("| axis | " + " | ".join(GAGES) + " |")
    P("|---|" + "---:|" * 6)
    for i in range(6):
        P(f"| {AXES[i]} | " + " | ".join(f"{C[i, j]:+.6g}" for j in range(6)) + " |")
    P("")
    P("## Residuals")
    P("")
    P("| axis | test range | rms | max | max % of test range" + (" | max % FS |" if rated else " |"))
    P("|---|---:|---:|---:|---:|" + ("---:|" if rated else ""))
    for i in range(6):
        u = "N" if i < 3 else "N.m"
        rng = np.ptp(L[:, i])
        mx = np.max(np.abs(E[:, i]))
        pct = mx / rng * 100 if rng > 0 else float("nan")
        row = f"| {AXES[i]} | {rng:.3f} {u} | {np.sqrt(np.mean(E[:, i]**2)):.4f} | {mx:.4f} | {pct:.3f}"
        if rated:
            row += f" | {mx / rated[AXES[i]] * 100:.3f}"
        P(row + " |")
    P("")
    P("## Formulas")
    P("")
    for i in range(6):
        P(f"    {AXES[i]} = " + " ".join(f"{C[i, j]:+.6g}*d{GAGES[j]}" for j in range(6)))
    P("")
    P("## C array (firmware / Jade Studio)")
    P("")
    P("```c")
    P("/* Mini58 on Jade: F[i] = sum_j C[i][j] * (V[j] - V0[j]);  V in volts at sensor output */")
    P("static const float MINI58_C[6][6] = {")
    for i in range(6):
        P("    {" + ", ".join(f"{C[i, j]:.8e}f" for j in range(6)) + "},  /* " + AXES[i] + " */")
    P("};")
    P("```")
    rep = "\n".join(lines)
    print(rep)
    out = {"matrix": C.tolist(), "rows": AXES, "cols": GAGES, "units": {"V": "volt at sensor output",
           "F": "N", "T": "N.m"}, "n_points": n, "condition_number": cond,
           "front_end_ratio": cfg["front_end_ratio"], "pga_gain": cfg["pga_gain"],
           "gage_sign": cfg["gage_sign"], "gage_columns": cfg["gage_columns"]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.splitext(a.out)[0] + "_report.md", "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print(f"\nmatrix saved to {a.out}")


# --------------------------------------------------------------------------- ATI .cal

UNIT_F = {"N": 1.0, "lbf": 4.4482216152605, "lb": 4.4482216152605, "kgf": 9.80665, "kN": 1000.0}
UNIT_T = {"N-m": 1.0, "Nm": 1.0, "N.m": 1.0, "N-mm": 1e-3, "lbf-in": 0.1129848290276167,
          "lb-in": 0.1129848290276167, "lbf-ft": 1.3558179483314, "kgf-cm": 0.0980665}


def read_ati_cal(path: str) -> dict:
    """Parse an ATI FTxxxxx.cal (XML). Returns matrix in N and N.m per volt of gage output."""
    tree = ET.parse(path)
    root = tree.getroot()
    info = {"serial": root.attrib.get("Serial"), "body": root.attrib.get("BodyStyle"),
            "family": root.attrib.get("Family")}
    cal = root.find(".//Calibration")
    if cal is None:
        cal = root
    info.update({k: cal.attrib.get(k) for k in ("PartNumber", "CalDate", "ForceUnits", "TorqueUnits",
                                                 "OutputRange", "OutputMode", "GainMultiplier",
                                                 "HWTempComp", "OutputBipolar")})
    axes = cal.findall(".//UserAxis")
    which = "UserAxis"
    if len(axes) < 6:
        axes = cal.findall(".//Axis")
        which = "Axis"
    if len(axes) < 6:
        sys.exit(f"{path}: could not find 6 UserAxis/Axis rows")
    rows, maxes = {}, {}
    for ax in axes:
        name = ax.attrib.get("Name")
        vals = [float(x) for x in ax.attrib.get("values", "").split()]
        if name in AXES and len(vals) >= 6:
            rows[name] = vals[:6]
            if "max" in ax.attrib:
                maxes[name] = float(ax.attrib["max"])
    if len(rows) != 6:
        sys.exit(f"{path}: found rows {list(rows)}; need all of {AXES}")
    M = np.array([rows[a] for a in AXES])
    fu = UNIT_F.get(info.get("ForceUnits") or "N")
    tu = UNIT_T.get(info.get("TorqueUnits") or "N-m")
    if fu is None or tu is None:
        sys.exit(f"{path}: unknown units {info.get('ForceUnits')} / {info.get('TorqueUnits')}")
    conv = np.array([fu] * 3 + [tu] * 3)
    M_si = M * conv[:, None]
    maxes_si = {k: v * conv[AXES.index(k)] for k, v in maxes.items()}
    info.update({"matrix_source": which, "matrix_native": M, "matrix_si": M_si, "rated_si": maxes_si})
    return info


def cmd_ati(a, cfg):
    info = read_ati_cal(a.cal)
    M = info["matrix_si"]
    print(f"{a.cal}: serial {info['serial']}  {info['body']}  {info['PartNumber']}  cal date {info['CalDate']}")
    print(f"  units {info['ForceUnits']}/{info['TorqueUnits']}, OutputRange {info['OutputRange']}, "
          f"GainMultiplier {info['GainMultiplier']}, HWTempComp {info['HWTempComp']}; using <{info['matrix_source']}> rows")
    if info.get("GainMultiplier") not in (None, "1", "1.0"):
        print("  NOTE: GainMultiplier is not 1. The matrix below is used as-is; the weight check (gain factor) will"
              " show if a scale is missing.")
    print("\nATI matrix (N and N.m per volt):")
    for i in range(6):
        print(f"  {AXES[i]} = " + " ".join(f"{M[i, j]:+.6g}*d{GAGES[j]}" for j in range(6)))
    if info["rated_si"]:
        print("  rated: " + ", ".join(f"{k} {v:g}" for k, v in info["rated_si"].items()))
    if not a.steps:
        return
    steps = []
    for p in a.steps:
        steps += read_steps(p, a.log)
    steps = warn_saturation(process_steps(steps, cfg, tare=a.tare))
    D = np.array([s.dV for s in steps])
    L = np.array([s.L for s in steps])
    Lhat = D @ M.T
    print(f"\n{'step':>5} {'mass':>6} | " + " ".join(f"{x:>9}" for x in AXES) + "   (measured with ATI matrix)")
    for s, lh in zip(steps, Lhat):
        print(f"{s.step:>5} {s.mass:>6g} | " + " ".join(f"{x:>+9.3f}" for x in lh))
        print(f"{'':>5} {'appl.':>6} | " + " ".join(f"{x:>+9.3f}" for x in s.L))
    # one scalar gain that best maps ATI output to applied loads (Jade gain/divider check)
    m = np.any(L != 0, axis=1)
    if m.any():
        w = np.array([1, 1, 1, a.arm_inv, a.arm_inv, a.arm_inv])
        num = np.sum((Lhat[m] * w) * (L[m] * w))
        den = np.sum((Lhat[m] * w) ** 2)
        k = num / den
        print(f"\nBest single gain correction factor: {k:.5f}  (1.00000 = Jade chain matches ATI's volts)."
              f"\nIf it is far from 1, check pga_gain / front_end_ratio in the config first.")
        E = Lhat * k - L
        for i in range(6):
            u = "N" if i < 3 else "N.m"
            print(f"  {AXES[i]}: max error after factor {np.max(np.abs(E[:, i])):.4f} {u}")


def cmd_loads(a, cfg):
    steps = read_steps(a.steps, a.log)
    g = float(cfg["g"])
    print(f"{'step':>5} {'ser':>4} {'mass':>7} {'dir':>16} {'r (mm)':>22} | " + " ".join(f"{x:>8}" for x in AXES))
    for s in steps:
        L = load_vector(s.mass, s.direction, s.r_mm, g)
        d = "(" + ",".join(f"{x:+.2f}" for x in s.direction) + ")"
        r = "(" + ",".join(f"{x:.1f}" for x in s.r_mm) + ")"
        print(f"{s.step:>5} {s.series:>4} {s.mass:>7g} {d:>16} {r:>22} | " + " ".join(f"{x:>+8.3f}" for x in L))


# --------------------------------------------------------------------------- rangecheck

def cmd_rangecheck(a, cfg):
    """Tare, one known weight, tare: projected mass at which the first ADC channel hits its limit."""
    log = read_log(a.log, cfg)
    pl = [(t0 + 1.0, t1 - 1.0) for t0, t1 in find_plateaus(log, min_hold=a.min_hold) if t1 - t0 > 3.0]
    if len(pl) < 2:
        sys.exit(f"found {len(pl)} stable holds; need at least 2 (unloaded, then loaded). Check the log.")
    means = [log.v_adc[(log.t >= t0) & (log.t <= t1)].mean(0) for t0, t1 in pl]
    # loaded hold = the one furthest from the first (unloaded) hold
    d = [np.linalg.norm(m - means[0]) for m in means]
    il = int(np.argmax(d))
    V0, V1 = means[0], means[il]
    lim = adc_limits(log, cfg, float(cfg["adc_limit_fraction"]))
    dirv = parse_dir(a.dir)
    L = load_vector(a.mass, dirv, np.array([a.x, a.y, a.z]), float(cfg["g"]))
    print(f"holds found: {len(pl)}; unloaded = hold 1 ({pl[0][0]:.1f}-{pl[0][1]:.1f} s), "
          f"loaded = hold {il+1} ({pl[il][0]:.1f}-{pl[il][1]:.1f} s)")
    print(f"load of {a.mass} kg: " + ", ".join(f"{AXES[i]} {L[i]:+.3f}" for i in range(6) if abs(L[i]) > 1e-9) + "\n")
    print(f"{'ch':<5}{'V0 (ADC)':>11}{'dV per kg':>13}{'max kg':>9}")
    kmax = []
    for j in range(6):
        per_kg = (V1[j] - V0[j]) / a.mass
        if abs(per_kg) < 1e-9:
            mk = float("inf")
        else:
            room = (lim[j] - V0[j]) if per_kg > 0 else (V0[j] + lim[j])
            mk = room / abs(per_kg)
        kmax.append(mk)
        print(f"{GAGES[j]:<5}{V0[j]:>10.4f}V{per_kg*1e3:>10.2f}mV{mk:>9.2f}")
    j = int(np.argmin(kmax))
    print(f"\nFirst channel to reach {cfg['adc_limit_fraction']*100:.0f} % of the ADC window: {GAGES[j]} at "
          f"about {kmax[j]:.2f} kg in this setup (same direction, same lever arm).")
    print(f"Suggested top mass for the cycles: <= {0.8*kmax[j]:.2f} kg.")
    if len(pl) >= 3:
        z = means[-1]
        print("zero return after removing the weight (uV): " + " ".join(f"{(z[i]-V0[i])*1e6:+.0f}" for i in range(6)))


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help="JSON config (default jade_mini58.json next to this script)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quicklook", help="per-channel stats of a log")
    q.add_argument("log")
    q.add_argument("--t0", type=float)
    q.add_argument("--t1", type=float)

    sg = sub.add_parser("segment", help="detect stable plateaus in a continuous log")
    sg.add_argument("log")
    sg.add_argument("-o", "--out", default="steps_draft.csv")
    sg.add_argument("--window", type=float, default=1.0, help="moving window for stability (s)")
    sg.add_argument("--k", type=float, default=4.0, help="stable if moving std < k x noise floor")
    sg.add_argument("--min-hold", type=float, default=5.0, help="minimum plateau length (s)")
    sg.add_argument("--merge-gap", type=float, default=2.0)
    sg.add_argument("--edge", type=float, default=1.0, help="seconds trimmed at each end of a plateau")
    sg.add_argument("--dir", default="+x", help="default 'dir' written in the draft")
    sg.add_argument("--plan", help="planned steps CSV (mass, dir, lever arm) to copy onto the plateaus in order")
    sg.add_argument("--plot", action="store_true")

    ax = sub.add_parser("axis", help="single-axis linearity / hysteresis analysis")
    ax.add_argument("steps", nargs="+")
    ax.add_argument("--axis", default="Fx", choices=AXES)
    ax.add_argument("--log", help="log file for steps that have no 'file'")
    ax.add_argument("--tare", default="first", choices=["first", "drift"])
    ax.add_argument("-o", "--out", default="out_axis")
    ax.add_argument("--no-plot", action="store_true")

    ft = sub.add_parser("fit", help="full 6x6 least-squares fit")
    ft.add_argument("steps", nargs="+")
    ft.add_argument("--log")
    ft.add_argument("--tare", default="drift", choices=["first", "drift"])
    ft.add_argument("--arm", type=float, default=0.1, help="typical lever arm (m) used to weigh torque vs force")
    ft.add_argument("--force", action="store_true", help="fit even if the loads do not cover all axes")
    ft.add_argument("-o", "--out", default="mini58_C.json")

    at = sub.add_parser("ati", help="apply an ATI FTxxxxx.cal matrix")
    at.add_argument("cal")
    at.add_argument("steps", nargs="*")
    at.add_argument("--log")
    at.add_argument("--tare", default="first", choices=["first", "drift"])
    at.add_argument("--arm-inv", type=float, default=10.0, help="weight of torque errors (1/m) in the gain fit")

    rcp = sub.add_parser("rangecheck", help="max mass before the ADC limit, from one small weight")
    rcp.add_argument("log")
    rcp.add_argument("--mass", type=float, required=True, help="the weight hung (kg)")
    rcp.add_argument("--dir", default="+x", help="gravity direction in sensor axes")
    rcp.add_argument("--x", type=float, default=0.0, help="hook point x (mm)")
    rcp.add_argument("--y", type=float, default=0.0)
    rcp.add_argument("--z", type=float, default=0.0)
    rcp.add_argument("--min-hold", type=float, default=5.0)

    ld = sub.add_parser("loads", help="print the load vector of each step")
    ld.add_argument("steps")
    ld.add_argument("--log")

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    {"quicklook": cmd_quicklook, "segment": cmd_segment, "axis": cmd_axis, "fit": cmd_fit,
     "ati": cmd_ati, "loads": cmd_loads, "rangecheck": cmd_rangecheck}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
