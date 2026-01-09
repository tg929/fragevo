#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮盘赌选择算法（加权随机选择）
============================
基于分子的适应度分数进行加权随机选择。
分数越好的分子被选中的概率越高，但所有分子都有被选中的可能。
"""
import numpy as np

def spin_roulette_selector(molecules_data, number_to_choose, selection_type="docking"):
    """
    基于轮盘赌算法选择分子。
    
    Args:
        molecules_data: 分子数据列表，每个元素格式为 [SMILES, name, docking_score]
        number_to_choose: 要选择的分子数量
        selection_type: 选择类型，"docking"表示基于对接分数选择
    
    Returns:
        选中的分子SMILES数组(numpy array)
    """
    if not isinstance(molecules_data, list) or len(molecules_data) == 0:
        return np.array([])
    
    if number_to_choose <= 0:
        return np.array([])
    
    # 确保有足够的分子可供选择
    if len(molecules_data) < number_to_choose:
        return np.array([mol[0] for mol in molecules_data])
    
    # 调整分数以计算选择概率
    adjusted_scores = adjust_scores(molecules_data, selection_type)
    
    # 计算选择概率
    total_score = sum(adjusted_scores)
    if total_score <= 0:
        # 如果总分数为0或负数，使用均匀分布
        probabilities = [1.0 / len(molecules_data)] * len(molecules_data)
    else:
        probabilities = [score / total_score for score in adjusted_scores]
    
    # 提取SMILES列表
    smiles_list = [mol[0] for mol in molecules_data]
    
    # 使用numpy进行加权随机选择（无替换）
    try:
        selected_smiles = np.random.choice(
            smiles_list, 
            size=number_to_choose, 
            replace=False, 
            p=probabilities
        )
        return selected_smiles
    except ValueError as e:
        # 如果概率有问题，回退到简单的随机选择
        selected_indices = np.random.choice(
            len(smiles_list), 
            size=number_to_choose, 
            replace=False
        )
        return np.array([smiles_list[i] for i in selected_indices])

def adjust_scores(molecules_data, selection_type):
    """
    调整分数以用于轮盘赌选择。
    
    Args:
        molecules_data: 分子数据列表
        selection_type: 选择类型（"docking" 或 "diversity"）
    
    Returns:
        调整后的分数列表
    """
    if selection_type == "docking":
        # 对于对接分数，分数越小（越负）越好
        # 提取对接分数（最后一列）
        raw_scores = [float(mol[-1]) for mol in molecules_data]
        
        # 找到最大值（最不好的分数）并添加偏移量
        max_score = max(raw_scores)
        offset = max_score + 0.1 if max_score > 0 else 0.1
        
        # 转换分数：越小的原始分数转换后越大
        # 使用指数函数放大差异
        adjusted_scores = []
        for score in raw_scores:
            adjusted_score = (offset - score) ** 2
            adjusted_scores.append(adjusted_score)
        
        return adjusted_scores
    
    elif selection_type == "diversity":
        # 对于多样性分数，分数越小越好（越独特）
        diversity_scores = [float(mol[-1]) for mol in molecules_data]
        
        # 使用倒数平方来调整分数
        adjusted_scores = []
        for score in diversity_scores:
            if score > 0:
                adjusted_score = 1.0 / (score ** 2)
            else:
                adjusted_score = 1.0  # 避免除零错误
            adjusted_scores.append(adjusted_score)
        
        return adjusted_scores
    
    else:
        raise ValueError(f"不支持的选择类型: {selection_type}") 