#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子过滤模块
"""
import sys
import os
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, List
from autogrow.operators.filter.filter_classes.filter_children_classes import (
    lipinski_strict_filter,
    lipinski_lenient_filter,
    ghose_filter,
    ghose_modified_filter,
    vande_waterbeemd_filter,
    mozziconacci_filter,
    pains_filter,
    nih_filter,
    brenk_filter,
)
import autogrow.operators.convert_files.gypsum_dl.gypsum_dl.MolObjectHandling as MOH
FILTER_CLASS_MAP = {
    "lipinski_strict": lipinski_strict_filter.LipinskiStrictFilter,
    "lipinski_lenient": lipinski_lenient_filter.LipinskiLenientFilter,
    "ghose": ghose_filter.GhoseFilter,
    "ghose_modified": ghose_modified_filter.GhoseModifiedFilter,
    "vande_waterbeemd": vande_waterbeemd_filter.VandeWaterbeemdFilter,
    "mozziconacci": mozziconacci_filter.MozziconacciFilter,
    "pains": pains_filter.PAINSFilter,
    "nih": nih_filter.NIHFilter,
    "brenk": brenk_filter.BRENKFilter,
}

class FilterExecutor:
    """
    一个封装了分子过滤完整流程的执行器类。
    """
    def __init__(self, config: Dict):
        """
        初始化执行器，并根据配置设置好所有过滤器。        
        Args:
            config: 配置参数字典。
        """
        self.config = config
        self.filters = self._init_filters()
        print(f"过滤器执行器初始化完成，已加载过滤器: {', '.join(self.filters.keys())}")
    def _init_filters(self) -> Dict:       
        filters = {'Structure': StructureCheckFilter()}#基本结构过滤
        filter_config = self.config.get('filter', {})
        for filter_key, filter_class in FILTER_CLASS_MAP.items():
            config_key = f"enable_{filter_key}"
            if filter_config.get(config_key, False):
                filter_name = filter_class.__name__
                filters[filter_name] = filter_class()        
        if filter_config.get('no_filters', False) or len(filters) <= 1:
            print("警告: 未启用任何药物化学过滤器，仅执行基本结构检查。")
            return {'Structure': StructureCheckFilter()}        
        return filters
    def run_filtering(self, molecules: List[str]) -> List[str]:
        """
        对一组分子执行过滤操作。        
        Args:
            molecules: 待过滤的SMILES字符串列表。        
        Returns:
            过滤后剩下的SMILES字符串列表。
        """
        if not molecules:
            print("输入分子列表为空，无需过滤。")
            return []
        print(f"开始过滤 {len(molecules)} 个分子...")        
        filtered_molecules = []
        for smi in molecules:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                continue            
            if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
                continue
            passed_all = True
            for filter_obj in self.filters.values():
                if not filter_obj.run_filter(mol):
                    passed_all = False
                    break            
            if passed_all:
                filtered_molecules.append(smi)
        print(f"过滤完成，保留了 {len(filtered_molecules)} 个分子 "
              f"({len(filtered_molecules) / len(molecules) * 100:.1f}%)。")        
        return filtered_molecules
class StructureCheckFilter:
    def run_filter(self, mol):
        return mol is not None
def init_filters_from_config(config: Dict):
    """
    从配置字典初始化过滤器集合    
    Args:
        config: 配置参数字典        
    Returns:
        dict: 过滤器字典
    """
    filters = {'Structure': StructureCheckFilter()}  
    filter_config = config.get('filter', {})
    # 遍历所有可用的过滤器，并根据配置启用它们；
    for filter_key, filter_class in FILTER_CLASS_MAP.items():
        config_key = f"enable_{filter_key}"
        if filter_config.get(config_key, False):
            filter_name = filter_class.__name__
            filters[filter_name] = filter_class()
  
    alternative_filters = filter_config.get('alternative_filters', None)
    if alternative_filters:
        for filter_info in alternative_filters:
            if isinstance(filter_info, list) and len(filter_info) == 2:
                name, path = filter_info
                sys.path.append(os.path.dirname(path))
                module_name = os.path.basename(path).replace('.py', '')
                filter_module = __import__(module_name)
                filter_class = getattr(filter_module, name)
                filters[name] = filter_class()
    if filter_config.get('no_filters', False) or len(filters) <= 1:
        print("不应用任何药物化学过滤器，仅进行基本结构检查")
        return {'Structure': StructureCheckFilter()}    
    return filters
def run_filter_operation(config: Dict, input_file: str, output_file: str) -> str:
    """
    执行分子过滤操作    
    Args:
        config: 配置参数字典
        input_file: 输入文件路径
        output_file: 输出文件路径        
    Returns:
        str: 过滤后的结果文件路径
    """    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)        
    
    with open(input_file, 'r') as f:
        population = [line.strip() for line in f if line.strip()]       
 
    filters = init_filters_from_config(config)
    print(f"\n使用的过滤器: {', '.join(filters.keys())}")       
    filtered = []
    filter_counters = {name: 0 for name in filters.keys()}    
    for smi in population:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            continue
        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            continue        
        passed_all = True
        for name, filter_obj in filters.items():
            if filter_obj.run_filter(mol):
                filter_counters[name] += 1
            else:
                passed_all = False
                break                
        if passed_all:
            filtered.append(smi)    
    print("\n各过滤器通过率:")
    total_valid = sum(1 for smi in population if Chem.MolFromSmiles(smi) is not None)
    for name, counter in filter_counters.items():
        if total_valid > 0:
            print(f"{name}: {counter}/{total_valid} ({counter/total_valid*100:.1f}%)")     
   
    with open(output_file, 'w') as f:
        f.write("\n".join(filtered))    
    print(f"\n过滤后的分子已保存到: {output_file}")    
    return output_file
def run_filter_simple(config: Dict, molecules: List[str]) -> List[str]:  
    executor = FilterExecutor(config)
    return executor.run_filtering(molecules)
def main():
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='filter')
    parser.add_argument('--smiles_file', type=str, required=True, help='--smiles_file')
    parser.add_argument('--output_file', type=str, required=True, help='--output_file')
    parser.add_argument('--config_file', type=str, default='fragevo/config_example.json', help='--config_file')
    
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / args.config_file
    if not config_path.exists():
        raise FileNotFoundError(f"无法加载配置文件 {args.config_file}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not os.path.exists(args.smiles_file):
        print(f"输入SMILES文件不存在: {args.smiles_file}")
        return
    
    print(f"开始分子过滤: {args.smiles_file} -> {args.output_file}")
    
    results_file = run_filter_operation(config, args.smiles_file, args.output_file)
    print(f"过滤完成，结果保存在: {results_file}")

if __name__ == "__main__":
    main()
