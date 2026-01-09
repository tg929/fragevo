#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_smi_file(file_path):
    """解析对接结果文件，返回分子列表和得分列表"""
    molecules = []
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    molecules.append(parts[0])
                    scores.append(float(parts[1]))
    return molecules, scores

def analyze_results(molecules, scores, output_dir, prefix):
    """分析结果并输出排序后的文件和统计信息"""
    # 按得分排序（从小到大）
    sorted_indices = np.argsort(scores)
    sorted_molecules = [molecules[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # 输出排序后的文件
    sorted_file = os.path.join(output_dir, f"{prefix}_sorted.smi")
    with open(sorted_file, 'w') as f:
        for mol, score in zip(sorted_molecules, sorted_scores):
            f.write(f"{mol}\t{score:.2f}\n")
    
    # 计算各top分段的平均得分
    top_ranges = [1, 10, 20, 50]
    stats = {}
    
    for top in top_ranges:
        if len(sorted_scores) >= top:
            avg_score = np.mean(sorted_scores[:top])
            stats[f"top{top}"] = avg_score
        else:
            stats[f"top{top}"] = "N/A (Not enough molecules)"
    
    # 输出统计信息
    stats_file = os.path.join(output_dir, f"{prefix}_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Total molecules: {len(sorted_scores)}\n")
        for top in top_ranges:
            f.write(f"Top {top} average score: {stats[f'top{top}']}\n")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analyze docking results')
    parser.add_argument('-i', '--input', required=True, 
                        help='Input SMILES file with docking scores')
    parser.add_argument('-o', '--output', default=os.path.join(PROJECT_ROOT, 'operations/ranking'),
                        help='Output directory for results')
    parser.add_argument('-p', '--prefix', default='result',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 解析文件并分析结果
    molecules, scores = parse_smi_file(args.input)
    stats = analyze_results(molecules, scores, args.output, args.prefix)
    
    # 打印简要统计信息
    
    print(f"Analysis completed for {args.input}")
    print("\ngeneration_2")
    print(f"Total molecules processed: {len(scores)}")
    for top in [1, 10, 20, 50]:
        print(f"Top {top} average score: {stats[f'top{top}']}")

if __name__ == "__main__":
    main()