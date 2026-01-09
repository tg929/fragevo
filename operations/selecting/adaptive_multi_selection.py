#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应多目标选择策略
根据当前种群分布动态调整选择压力
"""
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def analyze_population_distribution(molecules_data: List[Dict]) -> Dict[str, float]:
    """
    分析当前种群在目标空间的分布情况
    
    Args:
        molecules_data: 分子数据列表
        
    Returns:
        分布分析结果
    """
    if not molecules_data:
        return {}
    
    docking_scores = [m['docking_score'] for m in molecules_data]
    qed_scores = [m.get('qed_score', 0) for m in molecules_data]
    sa_scores = [m.get('sa_score', 10) for m in molecules_data]
    
    analysis = {
        'docking_mean': np.mean(docking_scores),
        'docking_std': np.std(docking_scores),
        'qed_mean': np.mean(qed_scores),
        'qed_std': np.std(qed_scores),
        'sa_mean': np.mean(sa_scores),
        'sa_std': np.std(sa_scores),
        'total_molecules': len(molecules_data)
    }
    
    # 计算各目标的覆盖范围
    analysis['docking_range'] = max(docking_scores) - min(docking_scores)
    analysis['qed_range'] = max(qed_scores) - min(qed_scores)
    analysis['sa_range'] = max(sa_scores) - min(sa_scores)
    
    # 识别薄弱区域
    analysis['needs_better_docking'] = analysis['docking_mean'] > -8  # 对接分数不够好
    analysis['needs_better_qed'] = analysis['qed_mean'] < 0.6        # QED分数不够好
    analysis['needs_better_sa'] = analysis['sa_mean'] > 4            # SA分数不够好
    
    return analysis

def calculate_adaptive_weights(population_analysis: Dict[str, float], 
                             generation: int, max_generations: int) -> Tuple[float, float, float]:
    """
    根据种群分析和进化代数计算自适应权重
    
    Args:
        population_analysis: 种群分布分析结果
        generation: 当前代数
        max_generations: 最大代数
        
    Returns:
        (docking_weight, qed_weight, sa_weight)
    """
    # 基础权重
    base_docking = 1.0
    base_qed = 0.5
    base_sa = 0.3
    
    # 进化阶段调整
    early_stage = generation < max_generations * 0.3
    mid_stage = max_generations * 0.3 <= generation < max_generations * 0.7
    late_stage = generation >= max_generations * 0.7
    
    if early_stage:
        # 早期：重点探索，平衡各目标
        docking_weight = base_docking
        qed_weight = base_qed * 1.2
        sa_weight = base_sa * 1.2
    elif mid_stage:
        # 中期：根据种群分布调整
        docking_weight = base_docking * (1.5 if population_analysis.get('needs_better_docking', False) else 1.0)
        qed_weight = base_qed * (1.5 if population_analysis.get('needs_better_qed', False) else 1.0)
        sa_weight = base_sa * (1.5 if population_analysis.get('needs_better_sa', False) else 1.0)
    else:
        # 后期：重点优化主要目标
        docking_weight = base_docking * 1.5
        qed_weight = base_qed * 0.8
        sa_weight = base_sa * 0.8
    
    # 归一化权重
    total_weight = docking_weight + qed_weight + sa_weight
    docking_weight /= total_weight
    qed_weight /= total_weight
    sa_weight /= total_weight
    
    logger.info(f"第{generation}代自适应权重: 对接={docking_weight:.3f}, QED={qed_weight:.3f}, SA={sa_weight:.3f}")
    
    return docking_weight, qed_weight, sa_weight

def adaptive_multi_objective_selection(molecules_data: List[Dict], n_select: int,
                                     generation: int = 1, max_generations: int = 10) -> List[Dict]:
    """
    自适应多目标选择主函数
    
    Args:
        molecules_data: 分子数据列表
        n_select: 要选择的分子数量
        generation: 当前代数
        max_generations: 最大代数
        
    Returns:
        选中的分子列表
    """
    if not molecules_data or n_select <= 0:
        return []
    
    # 分析当前种群分布
    analysis = analyze_population_distribution(molecules_data)
    
    # 计算自适应权重
    docking_weight, qed_weight, sa_weight = calculate_adaptive_weights(
        analysis, generation, max_generations
    )
    
    # 应用权重的目标矩阵
    objectives = np.array([
        [m['docking_score'] * docking_weight, 
         -m.get('qed_score', 0) * qed_weight, 
         m.get('sa_score', 10) * sa_weight] 
        for m in molecules_data
    ])
    
    # 执行NSGA-II选择
    from operations.selecting.selecting_multi_demo import non_dominated_sort, crowding_distance
    fronts = non_dominated_sort(objectives)
    
    selected_molecules = []
    
    for front in fronts:
        if len(selected_molecules) + len(front) <= n_select:
            selected_molecules.extend([molecules_data[i] for i in front])
        else:
            # 使用拥挤度距离选择
            distances = crowding_distance(objectives, front)
            sorted_by_crowding = sorted(zip(front, distances), 
                                      key=lambda x: x[1], reverse=True)
            
            remaining_needed = n_select - len(selected_molecules)
            for i in range(remaining_needed):
                if i < len(sorted_by_crowding):
                    idx = sorted_by_crowding[i][0]
                    selected_molecules.append(molecules_data[idx])
            break
    
    logger.info(f"自适应选择完成: 共选择 {len(selected_molecules)} 个分子")
    return selected_molecules

def load_molecules_with_scores_and_deduplicate(parent_file: str, docked_file: str):
    """
    加载并合并父代和子代分子数据，自动去重
    """
    from operations.selecting.selecting_multi_demo import load_molecules_with_scores
    
    # 加载父代分子
    parent_molecules = load_molecules_with_scores(parent_file)
    logger.info(f"从父代文件加载了 {len(parent_molecules)} 个分子")
    
    # 加载子代分子  
    offspring_molecules = load_molecules_with_scores(docked_file)
    logger.info(f"从子代文件加载了 {len(offspring_molecules)} 个分子")
    
    # 合并并去重：使用字典以SMILES为key，保留分数更好的分子
    molecules_dict = {}
    total_before_dedup = len(parent_molecules) + len(offspring_molecules)
    
    # 先添加父代分子
    for mol in parent_molecules:
        smiles = mol['smiles']
        score = mol['docking_score']
        if smiles not in molecules_dict or score < molecules_dict[smiles]['docking_score']:
            molecules_dict[smiles] = mol
    
    # 然后添加子代分子（保留更好分数的）
    for mol in offspring_molecules:
        smiles = mol['smiles']
        score = mol['docking_score']
        if smiles not in molecules_dict or score < molecules_dict[smiles]['docking_score']:
            molecules_dict[smiles] = mol
    
    all_molecules = list(molecules_dict.values())
    duplicates_removed = total_before_dedup - len(all_molecules)
    
    logger.info(f"合并前总数: {total_before_dedup}, 去重后: {len(all_molecules)}, 删除重复: {duplicates_removed}")
    logger.info(f"去重率: {duplicates_removed/total_before_dedup*100:.1f}%")
    
    return all_molecules

def main():
    import argparse
    from operations.selecting.selecting_multi_demo import add_additional_scores, save_selected_molecules_with_scores, print_selection_statistics
    
    parser = argparse.ArgumentParser(description="自适应多目标分子选择")
    parser.add_argument('--docked_file', type=str, required=True, help='子代对接结果文件')
    parser.add_argument('--parent_file', type=str, required=True, help='父代对接结果文件')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件')
    parser.add_argument('--n_select', type=int, required=True, help='选择数量')
    parser.add_argument('--generation', type=int, default=1, help='当前代数')
    parser.add_argument('--max_generations', type=int, default=10, help='最大代数')
    
    args = parser.parse_args()
    
    logger.info("开始自适应多目标分子选择...")
    logger.info(f"子代文件: {args.docked_file}")
    logger.info(f"父代文件: {args.parent_file}")
    logger.info(f"输出文件: {args.output_file}")
    logger.info(f"选择数量: {args.n_select}")
    
    # 1. 加载并去重分子数据
    molecules = load_molecules_with_scores_and_deduplicate(args.parent_file, args.docked_file)
    if not molecules:
        logger.error("错误: 无法加载分子数据")
        return
    
    # 2. 计算QED和SA分数
    molecules = add_additional_scores(molecules)
    
    # 3. 使用自适应多目标选择
    selected_molecules = adaptive_multi_objective_selection(
        molecules, args.n_select, args.generation, args.max_generations
    )
    
    # 4. 保存结果
    if selected_molecules:
        save_selected_molecules_with_scores(selected_molecules, args.output_file)
        print_selection_statistics(selected_molecules)
    else:
        logger.error("错误: 未选择任何分子")

if __name__ == "__main__":
    main()