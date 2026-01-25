#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建自定义初始种群脚本

本脚本根据特定策略从一个大的分子数据库如ZINC250k中筛选分子,
以构建一个具有特定分层结构的初始种群。

"""

import argparse
import random
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Tuple
import numpy as np
from rdkit.Chem import BRICS

# --- 核心分解函数 ---

dummy = Chem.MolFromSmiles('[*]')

def canonicalize(smi: str, clear_stereo: bool = False) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    if clear_stereo: Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def mol_to_smiles(mol: Chem.Mol) -> str:
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)

def mols_to_smiles(mols: List[Chem.Mol]) -> List[str]:
    return [mol_to_smiles(m) for m in mols]

def count_dummies(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)

def strip_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    hydrogen = Chem.MolFromSmiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    return Chem.RemoveHs(mols[0]) if mols else None

def fragment_recursive(mol: Chem.Mol, frags: List[Chem.Mol]) -> List[Chem.Mol]:
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
        if not bonds:
            frags.append(mol)
            return frags
        bond_idxs = sorted([mol.GetBondBetweenAtoms(a1, a2).GetIdx() for a1, a2 in [b[0] for b in bonds]])
        if not bond_idxs:
            frags.append(mol)
            return frags
        broken = Chem.FragmentOnBonds(mol, bondIndices=[bond_idxs[0]], dummyLabels=[(0, 0)])
        res = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False)
        if len(res) == 2:
            frags.append(res[0])
            return fragment_recursive(res[1], frags)
        else:
            frags.append(mol)
            return frags
    except Exception:
        frags.append(mol)
        return frags

def join_molecules(molA: Chem.Mol, molB: Chem.Mol) -> Chem.Mol:
    marked = next((atom for atom in molA.GetAtoms() if atom.GetAtomicNum() == 0), None)
    if marked is None: return None
    neigh = marked.GetNeighbors()[0] if marked.GetNeighbors() else None
    ed = Chem.EditableMol(molA)
    ed.RemoveAtom(marked.GetIdx())
    molA = ed.GetMol()
    neigh_idx = neigh.GetIdx() if neigh else 0
    if marked.GetIdx() < neigh_idx: neigh_idx -=1
    joined = Chem.ReplaceSubstructs(molB, dummy, molA, replacementConnectionPoint=neigh_idx, useChirality=False)[0]
    Chem.Kekulize(joined)
    return joined

def reconstruct(frags: List[Chem.Mol]) -> Tuple[Chem.Mol, List[Chem.Mol]]:
    if len(frags) == 1: return strip_dummy_atoms(frags[0]), frags
    if count_dummies(frags[0]) != 1 or count_dummies(frags[-1]) != 1: return None, None
    if any(count_dummies(f) != 2 for f in frags[1:-1]): return None, None
    mol = frags[0]
    for frag in frags[1:]:
        mol = join_molecules(mol, frag)
    mol_to_smiles(mol)
    return mol, frags

def break_into_fragments(mol: Chem.Mol, smi: str) -> Tuple[str, list, int, list]:
    frags_mol = fragment_recursive(mol, [])
    if not frags_mol or len(frags_mol) <= 1:
        return smi, [smi], 1, [mol]
    rec, frags_mol_ordered = reconstruct(frags_mol)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags_mol_ordered)
        return smi, fragments, len(fragments), frags_mol_ordered
    return smi, [smi], 1, [mol]


# --- 主逻辑 ---

def create_population(input_file, output_file, targets, seed):
    """根据策略创建初始种群的主函数"""
    
    print(f"从 '{input_file}' 读取分子...")
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 步骤2: 设置种子并随机打乱数据
    print(f"设置随机数种子为: {seed}")
    random.seed(seed) # <-- 新增：设置随机数种子

    print("随机打乱数据...")
    random.shuffle(lines)
    collected_molecules = {tier: [] for tier in targets}
    total_needed = sum(targets.values())
    
    print("开始筛选分子以构建初始种群...")
    pbar = tqdm(total=total_needed)
    for line in lines:
        if sum(len(v) for v in collected_molecules.values()) >= total_needed:
            print("\n所有层的分子均已收集完毕。")
            break

        line = line.strip()
        if not line:
            continue

        smi = line.split()[0]
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue

        _, _, num_frags, _ = break_into_fragments(mol, smi)
        if num_frags <= 1:
            continue
        
        mw = Descriptors.MolWt(mol)
        
        tier = None
        if mw <= 100: tier = "MW_up_to_100"
        elif 100 < mw <= 150: tier = "MW_100_to_150"
        elif 150 < mw <= 200: tier = "MW_150_to_200"
        elif 200 < mw <= 250: tier = "MW_200_to_250"
        
        if tier and len(collected_molecules[tier]) < targets[tier]:
            collected_molecules[tier].append(line)
            pbar.update(1)

    pbar.close()

    print(f"\n筛选完成。正在将结果写入 '{output_file}'...")
    with open(output_file, 'w') as f:
        total_written = 0
        for tier_name in targets.keys():
            molecules_in_tier = collected_molecules[tier_name]
            #f.write(f"# --- Tier: {tier_name} (需要: {targets[tier_name]}, 找到: {len(molecules_in_tier)}) ---\n")
            for molecule_line in molecules_in_tier:
                f.write(molecule_line + '\n')
            total_written += len(molecules_in_tier)
            #f.write("\n")
            
    print("--- 总结 ---")
    for tier_name, mol_list in collected_molecules.items():
        print(f"层级 {tier_name:>15}: {len(mol_list):>3} / {targets[tier_name]} 个分子")
    print("----------------")
    print(f"总计收集了 {total_written} / {total_needed} 个分子。")
    print(f"初始种群已成功保存到 '{output_file}'。")


def main():
    parser = argparse.ArgumentParser(description="根据特定策略创建可复现的初始种群。")
    parser.add_argument('-i', '--input', default='ZINC250k.smi', help='包含所有候选分子的输入SMILES文件。')
    parser.add_argument('-o', '--output', default='my_initial_population.smi', help='用于保存最终种群的输出文件。')
    parser.add_argument('--seed', type=int, default=42, help='用于随机数生成的种子，以确保结果可复现。')
    args = parser.parse_args()

    population_targets = {
        "MW_up_to_100": 40,
        "MW_100_to_150": 60,
        "MW_150_to_200": 80,
        "MW_200_to_250": 20
    }
    
    create_population(args.input, args.output, population_targets, args.seed)


if __name__ == '__main__':
    main()