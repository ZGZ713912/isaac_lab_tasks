"""Typed, cached views of the recorded angle-reference experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ('left_front', 'left_back', 'right_back', 'right_front')
FIELDS = ('physical_angle', 'physical_velocity', 'current_raw', 'command_current_raw',
          'command_prepare_time_s', 'feedback_time_s', 'command_sequence', 'feedback_sequence',
          'target_physical_angle', 'target_physical_velocity', 'feedback_age_s', 'temperature',
          'command_limited')
GLOBALS = ('host_time_s', 'elapsed_s', 'stage', 'actuation_active', 'frequency_hz')


def load_recording(path, cache_dir):
    path, cache_dir = Path(path), Path(cache_dir)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (digest + '.npz')
    if not cached.exists():
        columns = [f'/identification/{f}' for f in GLOBALS] + ['/chassis/imu/roll', '/chassis/imu/pitch']
        columns += [f'/chassis/{j}_joint/{f}' for j in JOINTS for f in FIELDS]
        frame = pd.read_csv(path, usecols=columns)
        arrays = {f: frame[f'/identification/{f}'].to_numpy() for f in GLOBALS}
        arrays['imu'] = frame[['/chassis/imu/roll', '/chassis/imu/pitch']].to_numpy()
        for field in FIELDS:
            arrays[field] = frame[[f'/chassis/{j}_joint/{field}' for j in JOINTS]].to_numpy()
        np.savez_compressed(cached, **arrays)
    arrays = dict(np.load(cached, allow_pickle=False))
    return {'path': str(path.resolve()), 'sha256': digest, 'arrays': arrays,
            'metadata': json.loads(Path(str(path) + '.json').read_text()),
            'status': json.loads(Path(str(path) + '.status.json').read_text())}
