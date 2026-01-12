#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取已按受体排序/截取后的 CSV（默认：results/smiles_output_300ranked_by_receptor.csv），
按受体分组并在组内按 docking_score 升序统计：
  - 全部（通常为 300 条）的均值/方差
  - 前 50% / 20% / 10% / 5% / 3% 的均值/方差（向上取整，至少 1 个）

用法示例：
  python results/result_analyse.py \
      --ranked_csv results/smiles_output_300ranked_by_receptor.csv \
      --stats_csv results/smiles_output_top300_stats.csv
"""
from pathlib import Path
import argparse
import math
import statistics
import csv


def load_ranked_records(ranked_csv: Path):
    if not ranked_csv.exists():
        raise FileNotFoundError(f"未找到输入文件: {ranked_csv}")

    records = []
    with ranked_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)        
        # 期望字段：receptor, experiment, best_docking_score, qed, sa, smiles, generation, file
        for row in reader:
            if not row:
                continue
            try:
                receptor = (row.get('receptor') or row.get('Receptor') or '').strip()
                experiment = (row.get('experiment') or row.get('Experiment') or '').strip()                
                score_str = (
                    row.get('best_docking_score')
                    or row.get('docking_score')
                    or row.get('score')
                    or row.get('DockingScore')
                    or row.get('Docking_Score')
                    or ''
                )
                score_str = score_str.strip()
                if score_str == '' or receptor == '':
                    continue
                score = float(score_str)               
                qed = row.get('qed'); sa = row.get('sa')
                qed = None if (qed is None or qed == '' or qed == 'NA') else float(qed)
                sa = None if (sa is None or sa == '' or sa == 'NA') else float(sa)
                smiles = (row.get('smiles') or '').strip()
                generation = (row.get('generation') or '').strip()
                file_path = (row.get('file') or row.get('path') or '').strip()
                records.append({
                    'receptor': receptor,
                    'experiment': experiment,
                    'score': score,
                    'qed': qed,
                    'sa': sa,
                    'smiles': smiles,
                    'generation': generation,
                    'file': file_path,
                })
            except Exception:                
                continue
    return records


def group_sort_and_limit(records, limit_per_receptor: int):
    receptors = sorted({r['receptor'] for r in records})
    selected = []
    for rec in receptors:
        group = [r for r in records if r['receptor'] == rec]
        group.sort(key=lambda x: x['score'])
        if limit_per_receptor > 0:
            group = group[:limit_per_receptor]
        selected.extend(group)
    return selected, receptors


def compute_stats_per_receptor(selected, receptors, fractions=(0.50, 0.20, 0.10, 0.05, 0.03)):
    rows = []
    lines = []
    lines.append("按受体统计(基于各自前N条)：")
    for rec in receptors:
        scores = [r['score'] for r in selected if r['receptor'] == rec]
        n = len(scores)
        if n == 0:
            continue
        scores.sort()
        mean_all = statistics.mean(scores)
        var_all = statistics.pvariance(scores) if n > 1 else 0.0
        lines.append(f"- 受体: {rec} | 计入样本 n={n} | 300条均值: {mean_all:.6f} | 方差: {var_all:.6f}")
        frac_stats = []
        for frac in fractions:
            k = max(1, math.ceil(n * frac))
            sub = scores[:k]
            m = statistics.mean(sub)            
            v = statistics.pvariance(sub) if k > 1 else 0.0
            frac_stats.append((frac, k, m, v))
            pct = int(frac * 100)
            lines.append(f"  · 前{pct:>2}% | n={k:>3} | 均值: {m:.6f} | 方差: {v:.6f}")
        
        rows.append([
            rec,
            n,
            f"{mean_all:.6f}",
            f"{var_all:.6f}",
            # 50%
            frac_stats[0][1], f"{frac_stats[0][2]:.6f}", f"{frac_stats[0][3]:.6f}",
            # 20%
            frac_stats[1][1], f"{frac_stats[1][2]:.6f}", f"{frac_stats[1][3]:.6f}",
            # 10%
            frac_stats[2][1], f"{frac_stats[2][2]:.6f}", f"{frac_stats[2][3]:.6f}",
            # 5%
            frac_stats[3][1], f"{frac_stats[3][2]:.6f}", f"{frac_stats[3][3]:.6f}",
            # 3%
            frac_stats[4][1], f"{frac_stats[4][2]:.6f}", f"{frac_stats[4][3]:.6f}",
        ])
    return rows, "\n".join(lines) + "\n"


def write_stats(rows, stats_path: Path):
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'receptor', 'count_selected', 'mean_300', 'var_300',
            'top50_count', 'top50_mean', 'top50_var',
            'top20_count', 'top20_mean', 'top20_var',
            'top10_count', 'top10_mean', 'top10_var',
            'top05_count', 'top05_mean', 'top05_var',
            'top03_count', 'top03_mean', 'top03_var',
        ])
        for row in rows:
            w.writerow(row)

def write_report(report: str, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='按受体统计 Top-N docking 结果')
    ap.add_argument(
        '--ranked_csv',
        type=str,
        default='results/smiles_output_300ranked_by_receptor.csv',
        help='输入CSV（已按受体筛选/排序后的 top-N 列表）',
    )
    ap.add_argument(
        '--stats_csv',
        type=str,
        default='results/smiles_output_top300_stats.csv',
        help='输出统计CSV',
    )
    ap.add_argument(
        '--report_txt',
        type=str,
        default='results/smiles_output_top300_stats.txt',
        help='输出控制台同款统计文本（便于直接查看/归档）',
    )
    ap.add_argument('--per_receptor_limit', type=int, default=300, help='每个受体统计的最大数量（默认300）')
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    ranked_arg = Path(args.ranked_csv)
    if ranked_arg.is_absolute():
        input_path = ranked_arg
    else:
        cwd_candidate = (Path.cwd() / ranked_arg).resolve()
        repo_candidate = (repo_root / ranked_arg).resolve()
        input_path = cwd_candidate if cwd_candidate.exists() else repo_candidate

    stats_arg = Path(args.stats_csv)
    stats_path = stats_arg if stats_arg.is_absolute() else (repo_root / stats_arg).resolve()

    report_path = None
    if args.report_txt:
        report_arg = Path(args.report_txt)
        report_path = report_arg if report_arg.is_absolute() else (repo_root / report_arg).resolve()

    records = load_ranked_records(input_path)
    if not records:
        print(f"未从 {input_path} 读取到有效记录。")
        return

    selected, receptors = group_sort_and_limit(records, args.per_receptor_limit)
    rows, report = compute_stats_per_receptor(selected, receptors)
    print(report, end="")
    write_stats(rows, stats_path)
    print(f"已写出统计结果: {stats_path}")
    if report_path is not None:
        write_report(report, report_path)
        print(f"已写出统计文本: {report_path}")


if __name__ == '__main__':
    main()
