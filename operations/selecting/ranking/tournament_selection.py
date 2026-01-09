#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锦标赛选择算法（竞赛选择）
========================
通过随机分组竞赛的方式选择分子。
每次从候选池中随机选择一定数量的分子进行竞赛，选出最优者。
这种方法在保持选择压力的同时，也给较差的个体一定的被选中机会。
"""
import random
import math
import copy

def run_tournament_selector(molecules_data, number_to_choose, tournament_size, 
                           column_idx_to_select, favor_most_negative=True):
    """
    基于锦标赛算法选择分子。
    
    Args:
        molecules_data: 分子数据列表，每个元素格式为 [SMILES, name, score1, score2, ...]
        number_to_choose: 要选择的分子数量
        tournament_size: 每次锦标赛的参与者比例(0.0-1.0)
        column_idx_to_select: 用于比较的列索引
        favor_most_negative: True表示分数越小越好,False表示分数越大越好
    
    Returns:
        选中的分子完整信息列表
    """
    if not isinstance(molecules_data, list) or len(molecules_data) == 0:
        return []
    
    if number_to_choose <= 0:
        return []
    
    # 确保有足够的分子可供选择
    if len(molecules_data) < number_to_choose:
        return molecules_data
    
    # 计算每次锦标赛的参与者数量
    num_participants = max(1, int(math.ceil(len(molecules_data) * tournament_size)))
    
    selected_molecules = []
    available_molecules = copy.deepcopy(molecules_data)
    
    # 进行number_to_choose次锦标赛
    for _ in range(number_to_choose):
        if not available_molecules:
            break
            
        # 运行一次锦标赛
        winner = run_single_tournament(
            available_molecules, 
            num_participants, 
            column_idx_to_select, 
            favor_most_negative
        )
        
        # 将获胜者添加到选中列表
        selected_molecules.append(winner)
        
        # 从可用列表中移除获胜者（无替换选择）
        available_molecules = [mol for mol in available_molecules if mol != winner]
    
    return selected_molecules

def run_single_tournament(molecules_data, num_participants, column_idx_to_select, 
                         favor_most_negative=True):
    """
    运行单次锦标赛选择。
    
    Args:
        molecules_data: 分子数据列表
        num_participants: 参与锦标赛的分子数量
        column_idx_to_select: 用于比较的列索引
        favor_most_negative: True表示分数越小越好，False表示分数越大越好
    
    Returns:
        获胜的分子信息
    """
    num_molecules = len(molecules_data)
    
    # 随机选择第一个参与者作为初始获胜者
    current_winner = molecules_data[random.randint(0, num_molecules - 1)]
    
    # 进行剩余的比较
    for _ in range(1, num_participants):
        # 随机选择一个挑战者
        challenger = molecules_data[random.randint(0, num_molecules - 1)]
        
        try:
            winner_score = float(current_winner[column_idx_to_select])
            challenger_score = float(challenger[column_idx_to_select])
            
            # 根据favor_most_negative参数决定比较方式
            if favor_most_negative:
                # 分数越小越好
                if challenger_score < winner_score:
                    current_winner = challenger
            else:
                # 分数越大越好
                if challenger_score > winner_score:
                    current_winner = challenger
                    
        except (ValueError, IndexError):
            # 如果分数无法转换或索引错误，跳过这次比较
            continue
    
    return current_winner 