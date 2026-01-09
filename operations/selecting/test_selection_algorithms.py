#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地选择算法测试脚本
====================
该脚本用于验证本地实现的三种分子选择算法的正确性：
- 排名选择 (Rank Selection)
- 轮盘赌选择 (Roulette Selection)
- 锦标赛选择 (Tournament Selection)

它使用一组预定义的模拟分子数据进行测试，并打印出每种算法的选择结果
以及预期的行为，以确保其功能符合遗传算法的要求。
"""

import sys
from pathlib import Path
import numpy as np

# 将项目根目录添加到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 从本地模块导入选择算法
from operations.selecting.ranking import rank_selection
from operations.selecting.ranking import roulette_selection
from operations.selecting.ranking import tournament_selection

# --- 模拟数据 ---
# 创建一个分子数据列表，格式: [SMILES, name, docking_score]
# 对接分数越低越好
MOCK_MOLECULES = [
    ['C(C)C',     'mol_best_1', -12.5],
    ['C(N)C',     'mol_best_2', -11.8],
    ['C1CCCCC1',  'mol_best_3', -11.8], # 与上一个分数相同
    ['C(O)C',     'mol_good_1', -11.5],
    ['CC(C)C',    'mol_mid_1',  -9.8],
    ['CC(N)C',    'mol_mid_2',  -9.5],
    ['CC(O)C',    'mol_mid_3',  -9.1],
    ['CCC(C)C',   'mol_bad_1',  -8.2],
    ['CCC(N)C',   'mol_bad_2',  -8.1],
    ['CCC(O)C',   'mol_worst',  -7.5],
]

def run_tests():
    """执行所有选择算法的测试"""
    
    number_to_select = 4

    print("=" * 60)
    print(f"开始测试选择算法，将从 {len(MOCK_MOLECULES)} 个分子中选择 {number_to_select} 个。")
    print("=" * 60)

    # --- 1. 测试排名选择 (Rank Selection) ---
    print("\n[1] --- 测试排名选择 (精英主义) ---")
    print("预期行为: 确定性地选择对接分数最低的4个分子。")
    
    selected_rank = rank_selection.run_rank_selector(
        molecules_data=MOCK_MOLECULES,
        number_to_choose=number_to_select,
        column_idx_to_select=-1,  # 按最后一列（对接分数）排序
        reverse_sort=False       # False表示分数越小越好
    )
    
    print(f"选择结果: {selected_rank}")
    expected_rank = ['C(C)C', 'C(N)C', 'C1CCCCC1', 'C(O)C']
    print(f"预期结果: {expected_rank}")
    
    # 验证结果
    if sorted(selected_rank) == sorted(expected_rank):
        print("测试通过 ✓")
    else:
        print("测试失败 ✗")

    # --- 2. 测试锦标赛选择 (Tournament Selection) ---
    print("\n[2] --- 测试锦标赛选择 ---")
    print("预期行为: 随机选择，但有很大概率选择分数较低的分子。每次运行结果可能不同。")

    selected_tournament = tournament_selection.run_tournament_selector(
        molecules_data=MOCK_MOLECULES,
        number_to_choose=number_to_select,
        tournament_size=0.2, # 每次从20%的池子中竞赛
        column_idx_to_select=-1,
        favor_most_negative=True
    )
    # 提取SMILES
    selected_tournament_smiles = [mol[0] for mol in selected_tournament]
    
    print(f"选择结果: {selected_tournament_smiles}")
    print("请检查结果是否倾向于选择分数较低的分子（如 -12.5, -11.8, -11.5）。")
    
    # 验证结果
    selected_scores = [mol[2] for mol in selected_tournament]
    if len(selected_tournament_smiles) == number_to_select and np.mean(selected_scores) < -9.5:
         print("测试通过 ✓ (结果具有随机性，但整体分数较低)")
    else:
        print("测试未通过 ✗ (检查逻辑或随机性)")


    # --- 3. 测试轮盘赌选择 (Roulette Selection) ---
    print("\n[3] --- 测试轮盘赌选择 ---")
    print("预期行为: 加权随机选择，分数越低的分子被选中的概率越高。每次运行结果可能不同。")

    selected_roulette = roulette_selection.spin_roulette_selector(
        molecules_data=MOCK_MOLECULES,
        number_to_choose=number_to_select,
        selection_type="docking"
    )
    
    print(f"选择结果: {selected_roulette.tolist()}")
    print("请检查结果是否倾向于选择分数较低的分子。")
    
    # 验证结果
    smiles_to_score = {mol[0]: mol[2] for mol in MOCK_MOLECULES}
    selected_scores_roulette = [smiles_to_score[s] for s in selected_roulette]
    if len(selected_roulette) == number_to_select and np.mean(selected_scores_roulette) < -9.5:
        print("测试通过 ✓ (结果具有随机性，但整体分数较低)")
    else:
        print("测试未通过 ✗ (检查逻辑或随机性)")


    print("\n" + "=" * 60)
    print("所有测试执行完毕。")
    print("=" * 60)


if __name__ == "__main__":
    run_tests() 