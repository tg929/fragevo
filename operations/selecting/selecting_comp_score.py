#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comp Score based selection script
============================================
This script selects next-generation parents using the scoring function:

    y = DS_hat * QED * SA_hat,   y in [0, 1]

where
    DS_hat = - clip(DS, [-20, 0]) / 20
    SA_hat = (10 - SA) / 9

Inputs are the docked files of parent and offspring. The script merges them,
computes QED/SA (via RDKit and sascorer), normalizes DS/SA, and selects the
top N molecules by y. The output file keeps a backward-compatible layout where
the 1st column is SMILES and the 2nd column is docking score (DS). Extra
columns are appended: QED, SA, COMP_SCORE.

Optional novelty/quality filters can be enabled via JSON config under
selection.comp_score_settings.novelty_filter.
"""

import argparse
import os
import json
import sys
from typing import List, Dict, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import QED

# SA scorer (same as other modules in this repo)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try operations/scoring first
SCORING_DIR = os.path.join(PROJECT_ROOT, 'operations', 'scoring')
if SCORING_DIR not in sys.path:
    sys.path.insert(0, SCORING_DIR)
calc_sa = None
try:
    from sascorer import calculateScore as calc_sa  # type: ignore
except Exception:
    # Fallback: use fragmlm/utils/sascorer.py (ships with fpscores.pkl.gz)
    ALT_SCORING_DIR = os.path.join(PROJECT_ROOT, 'fragmlm', 'utils')
    if ALT_SCORING_DIR not in sys.path:
        sys.path.insert(0, ALT_SCORING_DIR)
    try:
        from sascorer import calculateScore as calc_sa  # type: ignore
    except Exception:
        calc_sa = None


def read_config(config_file: Optional[str]) -> dict:
    if not config_file:
        return {}
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg
    except Exception:
        return {}


def load_molecules_with_scores(path: str) -> List[Dict]:
    molecules = []
    if not path:
        return molecules
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                parts = s.replace('\t', ' ').split()
                if len(parts) < 2:
                    continue
                smi = parts[0]
                try:
                    ds = float(parts[1])
                except ValueError:
                    continue
                molecules.append({'smiles': smi, 'docking_score': ds})
    except FileNotFoundError:
        pass
    return molecules


def compute_qed_sa(smiles: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        qed = float(QED.qed(mol))
        sa = float(calc_sa(mol)) if calc_sa else None
        return qed, sa
    except Exception:
        return None, None


def normalize_ds(ds: float, clip_min: float = -20.0, clip_max: float = 0.0) -> float:
    if ds < clip_min:
        ds = clip_min
    if ds > clip_max:
        ds = clip_max
    return -ds / 20.0


def normalize_sa(sa: float, sa_max_value: float = 10.0, sa_denominator: float = 9.0) -> float:
    # SA expected in roughly [1, 10], map to [0,1]
    return max(0.0, min(1.0, (sa_max_value - sa) / sa_denominator))


def build_fp(smiles: str, fp_type: str = 'morgan', radius: int = 2, nbits: int = 2048):
    try:
        from rdkit.Chem import rdMolDescriptors
        from rdkit import DataStructs
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if fp_type == 'morgan':
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return fp
    except Exception:
        return None


def tanimoto_max_similarity(query_smi: str, reference_smis: List[str], fp_type: str = 'morgan', radius: int = 2, nbits: int = 2048) -> float:
    try:
        from rdkit import DataStructs
    except Exception:
        return 0.0
    qfp = build_fp(query_smi, fp_type, radius, nbits)
    if qfp is None or not reference_smis:
        return 0.0
    max_sim = 0.0
    for rs in reference_smis:
        rfp = build_fp(rs, fp_type, radius, nbits)
        if rfp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(qfp, rfp)
        if sim > max_sim:
            max_sim = sim
    return float(max_sim)


def apply_optional_filters(cands: List[Dict], cfg: dict) -> List[Dict]:
    settings = cfg.get('selection', {}).get('comp_score_settings', {})
    nf = settings.get('novelty_filter', {})
    if not nf or not nf.get('enable', False):
        return cands

    sim_th = float(nf.get('similarity_threshold', 0.4))
    qed_min = float(nf.get('qed_min', 0.5))
    sa_max = float(nf.get('sa_max', 5.0))
    ds_active_median = nf.get('ds_active_median', None)
    ref_file = nf.get('reference_smiles_file', None)
    fp_type = nf.get('fp_type', 'morgan')
    fp_radius = int(nf.get('fp_radius', 2))
    fp_nbits = int(nf.get('fp_nbits', 2048))

    references = []
    if ref_file and os.path.exists(ref_file):
        try:
            with open(ref_file, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip().split()[0]
                    if s:
                        references.append(s)
        except Exception:
            references = []

    filtered = []
    for m in cands:
        qed = m.get('qed_score')
        sa = m.get('sa_score')
        ds = m.get('docking_score')
        if qed is None or sa is None:
            continue
        if qed < qed_min:
            continue
        if sa > sa_max:
            continue
        if ds_active_median is not None:
            try:
                if ds >= float(ds_active_median):
                    continue
            except Exception:
                pass
        if references:
            max_sim = tanimoto_max_similarity(m['smiles'], references, fp_type, fp_radius, fp_nbits)
            if max_sim >= sim_th:
                continue
        filtered.append(m)
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Comp Score based selection')
    parser.add_argument('--docked_file', required=True, help='--docked_file')
    parser.add_argument('--parent_file', required=False, help='--parent_file')
    parser.add_argument('--output_file', required=True, help='--output_file')
    parser.add_argument('--n_select', type=int, default=None, help='--n_select')
    parser.add_argument('--config_file', type=str, default=None, help='--config_file')
    args = parser.parse_args()

    cfg = read_config(args.config_file)
    settings = cfg.get('selection', {}).get('comp_score_settings', {})

    n_select = args.n_select if args.n_select is not None else int(settings.get('n_select', 100))
    norm_cfg = settings.get('normalization', {})
    ds_clip_min = float(norm_cfg.get('ds_clip_min', -20.0))
    ds_clip_max = float(norm_cfg.get('ds_clip_max', 0.0))
    sa_max_value = float(norm_cfg.get('sa_max_value', 10.0))
    sa_denominator = float(norm_cfg.get('sa_denominator', 9.0))

    # Load candidates
    cands = []
    if args.parent_file:
        cands.extend(load_molecules_with_scores(args.parent_file))
    cands.extend(load_molecules_with_scores(args.docked_file))

    # Deduplicate by SMILES keeping best docking score (lower is better) for y calculation baseline
    merged: Dict[str, Dict] = {}
    for m in cands:
        smi = m['smiles']
        if smi not in merged:
            merged[smi] = m
        else:
            # keep the one with better DS (smaller)
            if m['docking_score'] < merged[smi]['docking_score']:
                merged[smi] = m

    # Compute QED/SA and y
    enriched: List[Dict] = []
    for m in merged.values():
        qed, sa = compute_qed_sa(m['smiles'])
        if qed is None or sa is None:
            continue
        ds_hat = normalize_ds(m['docking_score'], ds_clip_min, ds_clip_max)
        sa_hat = normalize_sa(sa, sa_max_value, sa_denominator)
        y = float(ds_hat) * float(qed) * float(sa_hat)
        m2 = dict(m)
        m2['qed_score'] = float(qed)
        m2['sa_score'] = float(sa)
        m2['comp_score'] = y
        enriched.append(m2)

    # Apply optional novelty/quality filters
    enriched = apply_optional_filters(enriched, cfg)

    # --- Docking-score elitism (optional) ---
    # Keep top-k by docking score (lower is better), then fill the rest by Comp Score.
    elitism_cfg = settings.get('elitism', {})
    enable_elitism = bool(elitism_cfg.get('enable', True))
    dock_top_k = int(elitism_cfg.get('dock_top_k', 0))

    elites: List[Dict] = []
    remaining: List[Dict] = enriched

    if enable_elitism and dock_top_k > 0 and enriched:
        ds_sorted = sorted(enriched, key=lambda d: d['docking_score'])
        elites = ds_sorted[:min(dock_top_k, len(ds_sorted))]
        elite_smiles = set(m['smiles'] for m in elites)
        remaining = [m for m in enriched if m['smiles'] not in elite_smiles]

    # Sort remaining by Comp Score desc to fill up to n_select
    remaining.sort(key=lambda d: d['comp_score'], reverse=True)
    selected = elites + (remaining[:max(0, n_select - len(elites))] if n_select > 0 else remaining)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for m in selected:
            f.write(f"{m['smiles']}\t{m['docking_score']:.6f}\t{m['qed_score']:.6f}\t{m['sa_score']:.6f}\t{m['comp_score']:.6f}\n")

    # Optional stats file next to output
    if settings.get('write_stats', True):
        try:
            base_dir = os.path.dirname(args.output_file)
            gen_folder = os.path.basename(base_dir)
            stats_path = os.path.join(base_dir, f"{gen_folder}_comp_score_selection_stats.txt")
            import numpy as np
            ys = [m['comp_score'] for m in enriched]
            ds_list = [m['docking_score'] for m in enriched]
            with open(stats_path, 'w', encoding='utf-8') as sf:
                sf.write("Comp Score Selection Stats\n")
                if ys:
                    sf.write(f"count={len(ys)}\n")
                    sf.write(f"y mean={np.mean(ys):.6f}, y max={np.max(ys):.6f}, y min={np.min(ys):.6f}\n")
                if ds_list:
                    sf.write(f"DS mean={np.mean(ds_list):.6f}, DS best(min)={np.min(ds_list):.6f}\n")
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    main()
