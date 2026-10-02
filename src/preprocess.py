"""Read only the first 100 observations; external labels are loaded separately."""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

FILES = {1: '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         2: '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
         3: '2018-04-12_batchdata_updated_struct_errorcorrect.mat'}

@dataclass
class Cell:
    cell_id: str
    batch: int
    policy: str
    summary: pd.DataFrame
    curves: dict


def _vector(dataset):
    # Slice on disk BEFORE materializing, even for channels with thousands of cycles.
    if dataset.ndim != 2 or 1 not in dataset.shape:
        raise ValueError(f'Expected MATLAB vector, got {dataset.shape}')
    return np.asarray(dataset[:, :100] if dataset.shape[0] == 1 else dataset[:100, :]).ravel()


def load_data(directory):
    cells, excluded = [], []
    for batch, filename in FILES.items():
        with h5py.File(Path(directory) / filename, 'r') as f:
            b = f['batch']
            for i in range(len(b['cycles'])):
                cid = f'b{batch}c{i}'
                # Eligibility only: no external target values leave this loader.
                life = np.asarray(f[b['cycle_life'][i, 0]]).ravel()
                if len(life) != 1 or not np.isfinite(life[0]) or life[0] <= 0:
                    excluded.append({'cell_id': cid, 'reason': 'missing_or_invalid_target'})
                    continue
                s = f[b['summary'][i, 0]]
                vectors = {k: _vector(s[k]) for k in ['cycle', 'QDischarge', 'IR', 'Tavg', 'chargetime']}
                if len({len(v) for v in vectors.values()}) != 1:
                    raise ValueError(f'{cid}: summary length mismatch')
                frame = pd.DataFrame(vectors)
                frame = frame.loc[frame.cycle.between(1, 100)].copy()
                if frame.cycle.duplicated().any() or not np.equal(frame.cycle, np.floor(frame.cycle)).all():
                    raise ValueError(f'{cid}: invalid cycle labels')
                policy = ''.join(chr(int(x)) for x in np.asarray(f[b['policy_readable'][i, 0]]).ravel())
                c = f[b['cycles'][i, 0]]
                voltage = np.asarray(f[b['Vdlin'][i, 0]]).ravel()
                curves = {}
                for number in [10, 100]:
                    # Curve row aligns with summary cycle; never use a row beyond 100.
                    positions = np.flatnonzero(vectors['cycle'] == number)
                    if len(positions) == 1:
                        pos = int(positions[0])
                        assert pos < 100
                        if pos < len(c['Qdlin']):
                            curves[number] = (voltage.copy(), np.asarray(f[c['Qdlin'][pos, 0]]).ravel())
                cells.append(Cell(cid, batch, policy, frame, curves))
    assert len({c.cell_id for c in cells}) == len(cells)
    assert {b: sum(c.batch == b for c in cells) for b in FILES} == {1: 46, 2: 39, 3: 44}
    return cells, excluded


def load_targets(directory, ids):
    """Explicit label access, called for B2/B3 only AFTER final model is frozen."""
    result = {}
    for batch, filename in FILES.items():
        chosen = [cid for cid in ids if cid.startswith(f'b{batch}c')]
        if not chosen:
            continue
        with h5py.File(Path(directory) / filename, 'r') as f:
            for cid in chosen:
                i = int(cid.split('c')[1])
                result[cid] = float(np.asarray(f[f['batch']['cycle_life'][i, 0]]).ravel()[0])
    y = pd.Series(result, name='cycle_life').reindex(ids)
    assert y.index.tolist() == list(ids) and np.isfinite(y).all() and (y > 0).all()
    return y
