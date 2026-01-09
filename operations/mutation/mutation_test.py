#6-16:完全接受一个初始种群入口

import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import random
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import autogrow.operators.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_strict_filter import LipinskiStrictFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_filter import GhoseFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.vande_waterbeemd_filter import VandeWaterbeemdFilter

PARSER = argparse.ArgumentParser()
PARSER = argparse.ArgumentParser(description='GA mutation parameters')
PARSER.add_argument("--input_file", "-i",type=str, required=True, help="输入SMILES文件路径")
PARSER.add_argument("--output_file", "-o",type=str, default=os.path.join(PROJECT_ROOT, "output/generation_0_mutationed.smi"), help="输出文件路径")     
PARSER.add_argument("--max_mutations", type=int, default=2, help="每个父代最大变异尝试次数")

def main():
    args = PARSER.parse_args()
    
    # 加载初始数据集
    base_smiles = []
    with open(args.input_file, 'r') as f:
        base_smiles = [line.split()[0].strip() for line in f]    
    # 使用基础数据集作为初始种群
    initial_population = set(base_smiles) ##去重
    print(f"初始种群大小: {len(initial_population)}")    
    # 变异参数配置
    vars = {
        'rxn_library': 'all_rxns',
        'rxn_library_file': os.path.join(PROJECT_ROOT, 'autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json'),
        'function_group_library': os.path.join(PROJECT_ROOT, 'autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json'),
        'complementary_mol_directory': os.path.join(PROJECT_ROOT, 'autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir'),
        'filter_object_dict': {           
            'Structure_check': lambda mol: mol is not None
        },
        'max_time_mcs_thorough': 1,
        'gypsum_thoroughness': 3
    }

    # 初始化变异器
    rxn_library_vars = [
        vars['rxn_library'],
        vars['rxn_library_file'],
        vars['function_group_library'],
        vars['complementary_mol_directory']
    ]
    mutation_results = []
    
    # 执行变异
    with tqdm(total=len(initial_population), desc="Processing mutations") as pbar:
        for idx, parent in enumerate(initial_population):
            new_mutations = []
            click_chem = SmileClickClass.SmilesClickChem(rxn_library_vars, new_mutations, vars['filter_object_dict'])            
            # 尝试最多max_mutations次变异
            success = False
            for attempt in range(args.max_mutations):
                # 使用静音上下文管理器来抑制不必要的输出
                with suppress_stdout_stderr():
                    result = click_chem.run_smiles_click2(parent)

                if not result:
                    continue                
                # 新增过滤检查
                valid_results = []
                for smi in result:
                    try:
                        # 执行所有过滤器
                        if all([check(smi) for check in vars['filter_object_dict'].values()]):
                            valid_results.append(smi)
                            break  # 只要一个有效结果就停止检查
                    except:
                        continue                
                if valid_results:
                    # 只取第一个有效结果
                    chosen_smi = valid_results[0]
                    # 严格去重检查
                    if chosen_smi not in initial_population and chosen_smi not in mutation_results:
                        mutation_results.append(chosen_smi)
                        success = True
                        break  # 成功即停止尝试            
            pbar.update(1)
            pbar.set_postfix({'success_rate': f"{len(mutation_results)}/{idx+1}"})
    # 保存变异产生的新分子到主输出文件
    with open(args.output_file, 'w') as f:
        for smi in mutation_results:
            f.write(f"{smi}\n")    
    print(f"\n变异操作完成:")
    print(f"- 输入分子数: {len(initial_population)}")
    print(f"- 成功变异分子数: {len(mutation_results)}")
    #print(f"- 变异成功率: {len(mutation_results)/len(initial_population)*100:.1f}%")
    print(f"- 结果已保存到: {args.output_file}")    
if __name__ == "__main__":
    main()


