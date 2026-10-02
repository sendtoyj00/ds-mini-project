"""Deterministic, label-free feature generation from the design (pages 5/6/13)."""
import re
import numpy as np
import pandas as pd
from scipy.stats import theilslopes


def parse_policy(text):
    m = re.fullmatch(r'\s*([\d.]+)C\(([\d.]+)%\)-([\d.]+)C(?:-newstructure)?\s*', text, re.I)
    if not m:
        return (np.nan,) * 3, 'unparsed:' + text.strip().lower()
    a, soc, b = map(float, m.groups())
    if not (a > 0 and b > 0 and 0 < soc <= 100):
        return (np.nan,) * 3, 'invalid:' + text
    # Group full numeric policy, conservatively ignoring the structural tag.
    return (a, soc, b), f'{a:g}C({soc:g}%)-{b:g}C'


def align_curve(curve, cfg):
    if curve is None:
        return None, 'missing_curve'
    v, q = (np.asarray(x, dtype=float).ravel() for x in curve)
    if len(v) != len(q) or len(v) < 3:
        return None, 'length'
    valid = np.isfinite(v) & np.isfinite(q)
    if valid.mean() < cfg['curve_valid_share'] or valid.sum() < 3:
        return None, 'nonfinite'
    order = np.argsort(v[valid]); v, q = v[valid][order], q[valid][order]
    if np.any(np.diff(v) <= 0):
        return None, 'duplicate_voltage'
    lo, hi, n = cfg['voltage_grid']
    if v[0] > lo or v[-1] < hi:
        return None, 'range'
    # Include intervals crossing either boundary: interpolation must not bridge a large gap.
    gaps = np.diff(v)[(v[:-1] < hi) & (v[1:] > lo)]
    if np.any(gaps > cfg['curve_max_gap']):
        return None, 'voltage_gap'
    return np.interp(np.linspace(lo, hi, n), v, q), 'ok'


def median(x, minimum=1):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) >= minimum else np.nan


def capacity_features(qd, cfg):
    first = qd.loc[qd.index.to_series().between(2, 6)]
    slope = qd.loc[qd.index.to_series().between(10, 100)].dropna()
    return median(first, cfg['capacity_initial_min_count']), (float(theilslopes(slope.values, slope.index.values)[0])
        if len(slope) >= cfg['capacity_slope_min_count'] else np.nan)


def cell_features(cell, cfg):
    s = cell.summary.set_index('cycle').copy()
    assert s.index.min() >= 1 and s.index.max() <= 100
    # Initialization is identified from observed channels, not batch or lifetime.
    initialization = (s[['QDischarge', 'IR', 'Tavg', 'chargetime']] == 0).all(axis=1)
    s.loc[initialization, :] = np.nan
    qd = s.QDischarge.where(s.QDischarge > 0)
    ir = s.IR.where(s.IR > 0)
    temp = s.Tavg.where(s.Tavg != 0)
    rolling = qd.rolling(cfg['capacity_rolling_window'], center=True,
                         min_periods=cfg['capacity_rolling_min_periods']).median()
    jumps = ((qd - rolling).abs() / rolling.abs()) > cfg['capacity_jump_relative']
    initial, slope = capacity_features(qd, cfg)
    clean_initial, clean_slope = capacity_features(qd.mask(jumps), cfg)
    a, ra = align_curve(cell.curves.get(10), cfg)
    b, rb = align_curve(cell.curves.get(100), cfg)
    var = np.var(b - a, ddof=cfg['dq_variance_ddof']) if a is not None and b is not None else np.nan
    (c1, soc, c2), group = parse_policy(cell.policy)
    dq = b - a if a is not None and b is not None else None
    ct = s.chargetime.where(s.chargetime > 0)
    f = dict(log10_dq_var=np.log10(var) if np.isfinite(var) and var > 0 else np.nan,
             peak_c_rate=max(c1, c2), switch_soc=soc, qd_initial_median=initial,
             qd_slope_10_100=slope, tavg_median_100=median(temp), ir_median_100=median(ir),
             qd_initial_median_clean=clean_initial, qd_slope_10_100_clean=clean_slope,
             dq_min=float(np.min(dq)) if dq is not None else np.nan,
             c_rate_1=c1, c_rate_2=c2, c_rate_gap=c1-c2,
             t80_min=60 * (soc / 100 / c1 + (0.8 - soc / 100) / c2) if np.isfinite(c1) else np.nan,
             chargetime_median_100=median(ct),
             qd_change=median(qd.loc[qd.index.to_series().between(96,100)])-initial)
    f['effective_c_rate_80'] = 48 / f['t80_min'] if f['t80_min'] > 0 else np.nan
    f['qd_ratio'] = (initial+f['qd_change']) / initial if initial > 0 else np.nan
    audit = dict(cell_id=cell.cell_id, batch=cell.batch, policy=cell.policy, protocol_group=group,
                 max_feature_cycle=int(s.index.max()), initialization_rows=int(initialization.sum()),
                 invalid_ir_rows=int((s.IR.notna() & (s.IR <= 0)).sum()),
                 capacity_jump_count=int(jumps.sum()), curve10_status=ra, curve100_status=rb,
                 dq_status='ok' if np.isfinite(var) and var > 0 else 'invalid_variance')
    return f, audit


def prepare_features(cells, cfg):
    rows, audits = [], []
    for cell in cells:
        f, audit = cell_features(cell, cfg); rows.append({'cell_id':cell.cell_id, **f}); audits.append(audit)
    x = pd.DataFrame(rows).set_index('cell_id')
    meta = pd.DataFrame(audits).set_index('cell_id')
    assert x.index.is_unique and x.index.equals(meta.index)
    assert not np.isinf(x.to_numpy()).any()
    assert meta.max_feature_cycle.le(100).all()
    return x, meta


def build_feature_sets():
    m1 = ['log10_dq_var']; m2 = m1 + ['peak_c_rate', 'switch_soc']
    m3 = m2 + ['qd_initial_median', 'qd_slope_10_100']
    m4 = m3 + ['tavg_median_100', 'ir_median_100']
    clean = m2 + ['qd_initial_median_clean', 'qd_slope_10_100_clean']
    return {'M0': [], 'M1': m1, 'M2': m2, 'M3': m3, 'M4': m4,
            'M3_clean': clean, 'M4_clean': clean + m4[-2:]}
