#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排名选择算法（精英选择）
======================
基于分子的适应度分数进行排名,选择排名最高的N个分子。
这是一种精英选择策略，确保最优个体能够进入下一代。
"""

def run_rank_selector(molecules_data, number_to_choose, column_idx_to_select, reverse_sort=False):
    """
    基于指定列的分数对分子进行排名选择。    
    Args:
        molecules_data: 分子数据列表，每个元素格式为 [SMILES, name, score1, score2, ...]
        number_to_choose: 要选择的分子数量
        column_idx_to_select: 用于选择的列索引（-1表示最后一列,-2表示倒数第二列)
        reverse_sort: 是否降序排序(True表示分数越大越好,False表示分数越小越好)    
    Returns:
        选中的分子SMILES列表
    """
    if not isinstance(molecules_data, list) or len(molecules_data) == 0:
        return []    
    if number_to_choose <= 0:
        return []    
    # 确保有足够的分子可供选择
    if len(molecules_data) < number_to_choose:
        return [mol[0] for mol in molecules_data]    
    # 按指定列的分数排序
    try:
        sorted_molecules = sorted(
            molecules_data, 
            key=lambda mol: float(mol[column_idx_to_select]), 
            reverse=reverse_sort
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"无法按列 {column_idx_to_select} 排序: {e}")
    
    # 去除重复的SMILES，保留分数更好的
    unique_molecules = []
    seen_smiles = set()    
    for mol in sorted_molecules:
        smiles = mol[0]
        if smiles not in seen_smiles:
            unique_molecules.append(mol)
            seen_smiles.add(smiles)    
    # 选择前N个分子
    selected_molecules = unique_molecules[:number_to_choose]    
    # 返回SMILES列表
    return [mol[0] for mol in selected_molecules] 