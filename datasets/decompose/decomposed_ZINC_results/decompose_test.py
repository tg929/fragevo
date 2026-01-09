#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子片段分解工具
=================
本脚本用于将输入SMILES文件中的分子分解为片段。
其分解逻辑与 demo_frags.py 脚本保持一致。

主要功能:
- 从输入文件读取SMILES(格式:SMILES ID)。
- 将每个分子分解为其BRICS片段列表。
- 若分子无法分解，则其自身被视为单一片段。
- 将分解结果(SMILES 和 片段列表）输出到一个文件。
- 将分解后的片段转换为GPT输入格式([BOS]...[SEP]...[EOS]）并输出到另一个文件。
"""

import argparse
from tqdm import tqdm
from rdkit import Chem
from typing import List, Tuple
import numpy as np
from rdkit.Chem import BRICS
from copy import deepcopy

# 用于标记化学键断裂点的虚拟原子
dummy = Chem.MolFromSmiles('[*]')

def canonicalize(smi: str, clear_stereo: bool = False) -> str:
    """将SMILES字符串规范化。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def mol_to_smiles(mol: Chem.Mol) -> str:
    """将RDKit Mol对象转换为规范化的SMILES字符串。"""
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)

def mols_to_smiles(mols: List[Chem.Mol]) -> List[str]:
    """将Mol对象列表转换为SMILES字符串列表。"""
    return [mol_to_smiles(m) for m in mols]

def count_dummies(mol: Chem.Mol) -> int:
    """计算分子中的虚拟原子（[*]）数量。"""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count

def strip_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    """将分子中的虚拟原子替换为氢，然后移除它们。"""
    hydrogen = Chem.MolFromSmiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    if not mols:
        return None
    mol = Chem.RemoveHs(mols[0])
    return mol

def fragment_recursive(mol: Chem.Mol, frags: List[Chem.Mol]) -> List[Chem.Mol]:
    """使用BRICS规则递归地将分子分解为片段。"""
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))

        if not bonds:
            frags.append(mol)
            return frags

        idxs, _ = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond:
                bond_idxs.append(bond.GetIdx())
        
        if not bond_idxs:
            frags.append(mol)
            return frags

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]

        broken = Chem.FragmentOnBonds(mol, bondIndices=[bond_idxs[0]], dummyLabels=[(0, 0)])
        res = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False)

        if len(res) == 2:
            head, tail = res
            frags.append(head)
            return fragment_recursive(tail, frags)
        else:
            frags.append(mol)
            return frags
    except Exception:
        frags.append(mol)
        return frags


def join_molecules(molA: Chem.Mol, molB: Chem.Mol) -> Chem.Mol:
    """使用虚拟原子连接两个分子片段。"""
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            if atom.GetNeighbors():
                neigh = atom.GetNeighbors()[0]
            break
    
    neigh_idx = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        if neigh_idx > marked:
            neigh_idx -= 1
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh_idx,
        useChirality=False)[0]

    Chem.Kekulize(joined)
    return joined

def reconstruct(frags: List[Chem.Mol]) -> Tuple[Chem.Mol, List[Chem.Mol]]:
    """从片段重构分子以验证分解的正确性。"""
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    if count_dummies(frags[0]) != 1 or count_dummies(frags[-1]) != 1:
        return None, None

    for frag in frags[1:-1]:
        if count_dummies(frag) != 2:
            return None, None
    
    mol = join_molecules(frags[0], frags[1])
    for frag in frags[2:]:
        mol = join_molecules(mol, frag)

    mol_to_smiles(mol)
    return mol, frags
        
def break_into_fragments(mol: Chem.Mol, smi: str) -> Tuple[str, list, int, list]:
    """
    分解分子的主函数。如果无法分解或重构失败，则返回原始分子。
    """
    frags_mol = fragment_recursive(mol, [])
    
    if not frags_mol or len(frags_mol) <= 1:
        return smi, [smi], 1, [mol]

    rec, frags_mol_ordered = reconstruct(frags_mol)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags_mol_ordered)
        return smi, fragments, len(fragments), frags_mol_ordered

    return smi, [smi], 1, [mol]

def process_smiles_file(input_file: str, decomp_output_file: str, gpt_output_file: str):
    """
    读取SMILES文件,分解每个分子,并将结果写入两个输出文件。
    """
    success_count = 0
    with open(input_file, 'r') as f_in, \
         open(decomp_output_file, 'w') as f_decomp_out, \
         open(gpt_output_file, 'w') as f_gpt_out:

        lines = f_in.readlines()
        for line in tqdm(lines, desc='正在分解分子'):
            parts = line.strip().split()
            if not parts:
                continue
            smi = parts[0]

            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    f_decomp_out.write(f"{smi}\t无效的SMILES\n")
                    continue

                # 分解分子
                _, fragments_list, _, _ = break_into_fragments(mol, smi)
                
                # 将分解结果写入第一个输出文件
                f_decomp_out.write(f"{smi}\t{str(fragments_list)}\n")
                
                # 将片段连接为GPT格式
                sep_joined = '[SEP]'.join(fragments_list)
                f_gpt_out.write(f"[BOS]{sep_joined}[EOS]\n")
                success_count += 1

            except Exception as e:
                error_msg = f"处理时发生错误: {e}"
                f_decomp_out.write(f"{smi}\t{error_msg}\n")
                print(f"处理SMILES失败: {smi} | 错误: {str(e)}")
    
    print(f"\n处理完成。成功处理 {success_count}/{len(lines)} 个分子。")

def main():
    """主函数，用于解析命令行参数并启动处理流程。"""
    parser = argparse.ArgumentParser(
        description="分子片段分解工具,用于GPT数据预处理。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True, 
                        help='输入的SMILES文件。\n每行格式应为: SMILES_string ID_string')
    parser.add_argument('-o_decomp', required=True, 
                        help='用于存储分解片段列表的输出文件。\n格式: Original_SMILES\\t[\'frag1\', \'frag2\', ...]')
    parser.add_argument('-o_gpt', required=True, 
                        help='用于存储GPT格式片段的输出文件。\n格式: [BOS]frag1[SEP]frag2[EOS]')
    
    args = parser.parse_args()

    process_smiles_file(args.input, args.o_decomp, args.o_gpt)

if __name__ == '__main__':
    main()