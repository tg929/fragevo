from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import QED

from fragmlm.utils import sascorer


def compute_qed_sa(smiles: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute QED and SA (raw RDKit-SA score in [1,10], lower is better).

    Returns (qed, sa). If SMILES is invalid, returns (None, None).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = None
    try:
        sa = float(sascorer.calculateScore(mol))
    except Exception:
        sa = None
    return qed, sa


@dataclass
class ChemMetricCache:
    """
    Persistent cache for per-SMILES metrics to avoid recomputation across generations.
    """

    cache_path: Optional[Path]
    _data: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict, init=False)
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path = Path(self.cache_path)
        try:
            if self.cache_path.exists():
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
        except Exception:
            # Best-effort cache: never fail the run because of cache issues.
            self._data = {}

    def get(self, smiles: str) -> Tuple[Optional[float], Optional[float]]:
        entry = self._data.get(smiles)
        if not isinstance(entry, dict):
            return None, None
        return entry.get("qed"), entry.get("sa")

    def set(self, smiles: str, qed: Optional[float], sa: Optional[float]) -> None:
        self._data[smiles] = {"qed": qed, "sa": sa}
        self._dirty = True

    def get_or_compute(self, smiles: str) -> Tuple[Optional[float], Optional[float]]:
        if smiles in self._data:
            return self.get(smiles)
        qed, sa = compute_qed_sa(smiles)
        self.set(smiles, qed, sa)
        return qed, sa

    def flush(self) -> None:
        if self.cache_path is None or not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + f".tmp_{os.getpid()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp_path, self.cache_path)
            self._dirty = False
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
