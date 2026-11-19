# mini58_cal

Processing tools for the ATI Mini58 (bare TW transducer, six half-bridge gages) read by Jade (ADS131M08).
Procedure: "Mini58 on Jade — Fx calibration and hysteresis procedure" (team doc).

```
pip install numpy pandas matplotlib
```

## Before the first run

Edit `jade_mini58.json`:

- `gage_columns`: the six CSV columns carrying SG0..SG5, in that order
- `time_column`: time column in seconds (or set `sample_rate_hz` if there is none)
- `units`: `counts` (signed ADS131M08 code), `counts_u24`, `V`, `mV` or `uV`
- `pga_gain`: start at 1, raise per channel after quicklook / rangecheck, then keep fixed for the whole calibration
- `front_end_ratio`: 1 (bare transducer wired straight to Jade)
- `ref_column`: channel logging the shared Vexc/2 reference (optional)
- `exc_column`, `exc_ratio`, `exc_nominal`: channel logging the bridge excitation through a divider (e.g. CH7, ratio 3, nominal 2.5 V). Every sample is scaled to the nominal excitation (ratiometric)
- `calibration`: the label (SI-700-30, SI-1400-60, SI-2800-120) to get errors in % FS

## Commands

| Command | What it does |
|---|---|
| `quicklook LOG.csv|JADE0001.BIN [--t0 --t1]` | offset, noise and ADC headroom per channel |
| `rangecheck LOG.csv --mass 2 --dir +x --z 40` | log = tare, one weight, tare. Prints the mass at which the first channel hits 95 % of the ADC window |
| `segment LOG.csv --plan steps_fx_A.csv -o steps_A.csv --plot` | finds the stable holds and copies the planned masses onto them in order |
| `axis steps_A.csv steps_B.csv --axis Fx -o out_fx` | slopes, R², nonlinearity, hysteresis, zero return, repeatability, plots, interim formula |
| `fit steps_*.csv -o mini58_C.json` | full 6x6 matrix C (refuses if the load cases do not cover all six axes) |
| `ati FT12345.cal [steps...]` | prints the ATI matrix in N and N·m per volt; with steps, compares to the applied loads and gives a gain factor |
| `loads steps.csv` | prints the load vector computed for each step (check signs before testing) |

Jade SD logs (`.BIN`) are read directly: COBS frames, CRC16 checked, PGA gains taken from the config frame in the file. Any command that takes a log accepts a .BIN.

## Steps file columns

`step, series, file, t_start, t_end, mass_kg, dir, x_mm, y_mm, z_mm, note`

- `series`: each series has its own tare (its first step with `mass_kg = 0`)
- `file`: log file; empty = the `--log` argument. One file per step works too (leave `t_start`/`t_end` empty, the first 3 s and last 1 s are dropped)
- `dir`: gravity direction in sensor axes, `+x`, `-x`, `+y`, `-y`, `+z`, `-z` or `vx;vy;vz`
- `x_mm, y_mm, z_mm`: hook point relative to the chosen origin. Torque = r × F

## Conventions

- All voltages are at the sensor output: V_sensor = V_adc × front_end_ratio (+ the measured reference if `ref_column` is set)
- Loads in N and N·m, g = 9.8066 m/s² (config)
- F = C · (V − V0)
