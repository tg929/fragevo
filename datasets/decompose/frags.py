#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
片段分解和掩码工具
=====================
用于将分子分解成片段并应用灵活的掩码策略,为GPT生成提供输入。

主要功能：
- 分子片段分解
- 灵活的片段掩码(支持掩码末尾n个片段)
- 批量处理分子文件
"""

# import os
# os.chdir("./datasets/decompose")
import argparse
from tqdm import tqdm  # 用于显示进度条
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Lipinski
from collections import defaultdict
from typing import List, Tuple, Optional
import numpy as np
from rdkit.Chem import BRICS
from copy import deepcopy

dummy = Chem.MolFromSmiles('[*]')

def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)

def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol

def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res

def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count

def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]
    #return [Chem.MolToSmiles(m, isomericSmiles=True, allBondsExplicit=True) for m in mols]


def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def fragment_recursive(mol, frags):
    bonds = list(BRICS.FindBRICSBonds(mol))

    if bonds == []:
        frags.append(mol)
        return frags

    idxs, _labs = list(zip(*bonds))

    bond_idxs = []
    for a1, a2 in idxs:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        bond_idxs.append(bond.GetIdx())

    order = np.argsort(bond_idxs).tolist()
    bond_idxs = [bond_idxs[i] for i in order]

    # 只会断开一根键，也就是说，如果某个片段可以切割两个断点，但是只会切割其中一个，另一个会跟该变短视作一个整体
    broken = Chem.FragmentOnBonds(
        mol,
        bondIndices=[bond_idxs[0]],
        dummyLabels=[(0, 0)],
    )
    frags_tuple = Chem.GetMolFrags(broken, asMols=True)
    
    if len(frags_tuple) == 2:
        head, tail = frags_tuple
        frags.append(head)
        return fragment_recursive(tail, frags)
    elif len(frags_tuple) == 1:
        # 如果切不断（比如在环上），就尝试下一根键
        if len(bond_idxs) > 1:
             # 这里简单递归处理，实际上最好是循环试
             # 为简单起见，如果切不断，就把原分子作为终点加入
             frags.append(mol)
             return frags
        else:
             frags.append(frags_tuple[0])
             return frags
    else:
        # 如果切成了多于2段（极少情况），取第一段为head，剩下合并或者仅取第二段为tail
        # 通常 BRICS 单键切断只会有2段。如果有更多，可能是复杂结构
        head = frags_tuple[0]
        frags.append(head)
        # 简单处理：把剩下的作为 tail 继续递归
        # 注意：这里可能需要把剩下的拼起来，或者只是取最大的一个作为 tail
        # 这是一个简单的容错处理
        if len(frags_tuple) > 1:
             # 合并剩余片段作为 tail (这比较复杂因为没有连接信息)
             # 或者，我们假设第二大的是 tail
             tail = frags_tuple[1] 
             return fragment_recursive(tail, frags)
        return frags

def join_molecules(molA, molB):
    """Join molA into molB by replacing a dummy atom in molB with molA.

    Returns None if dummy/neighbor topology is invalid.
    """
    marked, neigh = None, None

    # Find dummy atom (atomic num 0) in molA and its (only) neighbor.
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neighbors = list(atom.GetNeighbors())
            if not neighbors:
                # Broken fragment: dummy atom is disconnected.
                return None
            neigh = neighbors[0]
            break

    if marked is None:
        return None

    neigh_idx = 0 if neigh is None else neigh.GetIdx()

    # Remove dummy atom from molA; adjust neighbor index if needed.
    ed = Chem.EditableMol(molA)
    if neigh_idx > marked:
        neigh_idx -= 1
    ed.RemoveAtom(marked)
    molA = ed.GetMol()

    try:
        joined = Chem.ReplaceSubstructs(
            molB,
            dummy,
            molA,
            replacementConnectionPoint=neigh_idx,
            useChirality=False,
        )[0]
        Chem.Kekulize(joined, clearAromaticFlags=True)
        return joined
    except Exception:
        return None

def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    if count_dummies(frags[0]) != 1:
        return None, None

    if count_dummies(frags[-1]) != 1:
        return None, None

    for frag in frags[1:-1]:
        if count_dummies(frag) != 2:
            return None, None
    
    mol = join_molecules(frags[0], frags[1])
    if mol is None:
        return None, None

    for frag in frags[2:]:
        mol = join_molecules(mol, frag)
        if mol is None:
            return None, None

    # see if there are kekulization/valence errors
    try:
        mol_to_smiles(mol)
    except Exception:
        return None, None

    return mol, frags
        
def break_into_fragments(mol, smi):
    frags = []
    frags = fragment_recursive(mol, frags)

    if len(frags) == 0:
        return smi, np.nan, 0,[]

    if len(frags) == 1:
        return smi, smi, 1,frags

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags)
        return smi, fragments, len(frags), frags  # 直接返回列表

    return smi, [smi], 0, []  # 返回原始分子作为列表

def apply_flexible_masking(fragments_list: List[str], n_fragments_to_mask: int = 1) -> str:
    """
    对片段列表应用灵活的掩码策略    
    Args:
        fragments_list: 分子片段列表
        n_fragments_to_mask: 要掩码的片段数量（从末尾开始）        
    Returns:
        掩码后的片段序列，格式为 [BOS]frag1[SEP]frag2[SEP]...[SEP]
    """
    if not fragments_list or len(fragments_list) <= n_fragments_to_mask:
        # 如果片段数不足或要掩码的数量过多，返回空的开始标记
        return "[BOS][SEP]"    
    # 保留前面的片段，去掉末尾的n个片段
    visible_fragments = fragments_list[:-n_fragments_to_mask]    
    # 构建掩码序列
    masked_sequence = "[BOS]" + "[SEP]".join(visible_fragments) + "[SEP]"    
    return masked_sequence

def calculate_dynamic_mask_count(fragments_list: List[str], current_generation: int, max_generations: int) -> int:
    """
    根据当前代数和总代数动态计算需要掩码的片段数量    
    核心策略：
    - 早期代数(第1代):最大化多样性,只保留第一个片段作为起始点
    - 后期代数:逐渐减少掩码数量,允许GA进行更精细的优化    
    Args:
        fragments_list: 分子片段列表
        current_generation: 当前代数(从1开始)
        max_generations: 总代数        
    Returns:
        要掩码的片段数量
    """
    if not fragments_list or len(fragments_list) <= 1:
        return 0  # 如果片段不足，无法进行有效掩码    
    total_fragments = len(fragments_list)    
    # 第一代：最大化多样性，只保留第一个片段（掩码所有其他片段）
    if current_generation == 1:
        return total_fragments - 1    
    # 如果只有一代，返回默认值
    if max_generations <= 1:
        return 1
    
    # 线性衰减策略：从最大掩码数逐渐减少到最小掩码数
    initial_mask_fragments = total_fragments - 1  # 早期掩码数（只保留第一个片段）
    final_mask_fragments = 1  # 后期掩码数（保留大部分片段）
    
    # 确保final_mask_fragments不超过可用片段数
    final_mask_fragments = min(final_mask_fragments, total_fragments - 1)
    
    # 如果初始和最终掩码数相同，直接返回
    if initial_mask_fragments <= final_mask_fragments:
        return final_mask_fragments
    
    # 计算进度比例（0到1之间）
    progress = (current_generation - 1) / (max_generations - 1)
    
    # 线性插值计算当前代数的掩码数
    current_mask_count = initial_mask_fragments - (progress * (initial_mask_fragments - final_mask_fragments))
    
    # 四舍五入并确保在合理范围内
    mask_count = int(round(current_mask_count))
    mask_count = max(final_mask_fragments, min(mask_count, initial_mask_fragments))    
    return mask_count

def decompose_and_mask_molecules_dynamic(input_file: str, output_file: str, current_generation: int, max_generations: int) -> int:
    """
    分解分子并应用动态掩码的主函数,用于FragEvo工作流的动态掩码功能    
    Args:
        input_file: 输入SMILES文件路径
        output_file: 输出掩码片段文件路径
        current_generation: 当前代数
        max_generations: 总代数        
    Returns:
        成功处理的分子数量
    """
    success_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc=f'第{current_generation}代动态掩码处理'):
            smiles = line.strip().split()[0]  # 读取SMILES            
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # 分解分子
            _, fragments_list, num_frags, _ = break_into_fragments(mol, smiles)

            # 只处理能够分解的分子
            if isinstance(fragments_list, list) and num_frags > 1:
                # 使用动态掩码计算
                n_fragments_to_mask = calculate_dynamic_mask_count(
                    fragments_list, current_generation, max_generations
                )
                # 应用灵活掩码
                masked_sequence = apply_flexible_masking(fragments_list, n_fragments_to_mask)
                f_out.write(f"{masked_sequence}\n")
                success_count += 1
    return success_count

def decompose_and_mask_molecules(input_file: str, output_file: str, n_fragments_to_mask: int = 1) -> int:
    """
    分解分子并应用掩码的主函数,用于FragEvo工作流    
    Args:
        input_file: 输入SMILES文件路径
        output_file: 输出掩码片段文件路径
        n_fragments_to_mask: 要掩码的片段数量        
    Returns:
        成功处理的分子数量
    """
    success_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc=f'应用{n_fragments_to_mask}片段掩码'):
            smiles = line.strip().split()[0]  # 读取SMILES            
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # 分解分子
            _, fragments_list, num_frags, _ = break_into_fragments(mol, smiles)

            # 只处理能够分解的分子
            if isinstance(fragments_list, list) and num_frags > n_fragments_to_mask:
                # 应用灵活掩码
                masked_sequence = apply_flexible_masking(fragments_list, n_fragments_to_mask)
                f_out.write(f"{masked_sequence}\n")
                success_count += 1
    
    return success_count

def batch_process(input_file, output_file,output_file2,output_file3,output_file4):
    """批量处理文件的核心函数"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out,open(output_file2,"w") as f_out2,open(output_file3,"w") as f_out3,open(output_file4, "w") as f_out4:
        for line in tqdm(f_in, desc='Processing molecules'):
            smi = line.strip().split()[0]  # 读取每行第一个SMILES
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                f_out.write(f"{smi}\tInvalid\n")
                f_out2.write("Invalid\n")
                continue

            # 使用现有分解逻辑
            _, fragments_list, num_frag, _ = break_into_fragments(mol, smi)
            f_out.write(f"{smi}\t{str(fragments_list)}\n")
            if isinstance(fragments_list, list) and num_frag > 1:
                sep_joined = '[SEP]'.join(fragments_list)
                f_out2.write(f"[BOS]{sep_joined}[EOS]\n")
                f_out4.write(f"{smi}\n")  # 写入原始SMILES
                if len(fragments_list) > 1:
                    truncated_list = fragments_list[:-1]
                    truncated_joined = '[SEP]'.join(truncated_list)
                    f_out3.write(f"[BOS]{truncated_joined}[SEP]\n")

def main():
    """主函数，用于解析命令行参数和启动分子处理流程"""
    parser = argparse.ArgumentParser(description="分子片段分解和掩码工具")
    parser.add_argument('-i', '--input', help='输入SMILES文件')
    parser.add_argument('-o', '--output', help='输出SMILES文件')
    parser.add_argument('-o2', '--output2', help='输出片段文件')
    parser.add_argument('-o3', '--output3', help='输出掩码后的片段文件')
    parser.add_argument('-o4', '--output4', help='输出重构后的SMILES文件')
    
    # --- 新增和修改的参数 ---
    parser.add_argument('--mask_fragments', type=int, default=1, 
                        help='要掩码的片段数量（从末尾开始），用于固定掩码模式')
    parser.add_argument('--current_generation', type=int, default=None, 
                        help='当前进化代数，用于动态掩码计算')
    parser.add_argument('--max_generations', type=int, default=None, 
                        help='最大进化代数，用于动态掩码计算')
    parser.add_argument('--enable_dynamic_masking', action='store_true',
                        help='启用动态掩码模式。如果设置，将忽略 --mask_fragments')

    args = parser.parse_args()

    # --- 主逻辑分支 ---
    if args.enable_dynamic_masking:
        # 动态掩码模式
        if args.current_generation is None or args.max_generations is None:
            raise ValueError("动态掩码模式需要 --current_generation 和 --max_generations 参数。")
        print("启用动态掩码模式...")
        decompose_and_mask_molecules_dynamic(args.input, args.output3, args.current_generation, args.max_generations)
    
    elif args.output3 and args.mask_fragments >= 0:
        # 固定掩码模式
        print(f"启用固定掩码模式，掩码片段数: {args.mask_fragments}")
        decompose_and_mask_molecules(args.input, args.output3, args.mask_fragments)
        
    else:
        # 默认的批量处理模式
        print("未指定掩码模式，执行默认的批量分解处理...")
        batch_process(args.input, args.output, args.output2, args.output3, args.output4)

if __name__ == '__main__':
    main()
