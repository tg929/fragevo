import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from tdc import Evaluator, Oracle  
import random
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import autogrow.operators.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_strict_filter import LipinskiStrictFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_filter import GhoseFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.vande_waterbeemd_filter import VandeWaterbeemdFilter

PARSER = argparse.ArgumentParser()
PARSER = argparse.ArgumentParser(description='GA mutation parameters')
PARSER.add_argument("--input_file", "-i",type=str, required=True)#第一次：/data1/tgy/GA_llm/output/generation_crossover_0.smi
PARSER.add_argument("--llm_generation_file", "-l",type=str, default=os.path.join(PROJECT_ROOT, "fragmlm/output/test0/crossovered0_frags_new_0.smi"))
PARSER.add_argument("--output_file", "-o",type=str, default=os.path.join(PROJECT_ROOT, "output/generation_0_mutationed.smi"))     
PARSER.add_argument("--mutation_attempts", type=int, default=1)
PARSER.add_argument("--max_mutations", type=int, default=2, help="每个父代最大变异尝试次数")

# 初始化评估器（与交叉模块保持一致）
def init_evaluators():
    try:
        div_evaluator = Evaluator(name='Diversity')
        nov_evaluator = Evaluator(name='Novelty') 
        qed_evaluator = Oracle(name='qed')
        sa_evaluator = Oracle(name='sa')
        return div_evaluator, nov_evaluator, qed_evaluator, sa_evaluator
    except ImportError:
        print("请先安装TDC包:pip install tdc")
        exit(1)
# 评估函数（与交叉模块保持一致）
def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    # 添加空列表保护
    if len(smiles_list) == 0:
        return {
            'diversity': 0.0,
            'novelty': 0.0,
            'avg_qed': 0.0,
            'avg_sa': 0.0,
            'num_valid': 0
        }
        
    # 计算多样性时需要至少2个样本
    diversity = div_eval(smiles_list) if len(smiles_list)>=2 else 0.0
    
    # 计算新颖性时处理分母为零的情况
    try:
        novelty = nov_eval(smiles_list, ref_smiles)
    except ZeroDivisionError:
        novelty = 0.0
    
    results = {
        'diversity': diversity,
        'novelty': novelty,
        'avg_qed': np.mean([qed_eval(s) for s in smiles_list]) if smiles_list else 0.0,
        'avg_sa': np.mean([sa_eval(s) for s in smiles_list]) if smiles_list else 0.0,
        'num_valid': len(smiles_list)
    }
    return results
def main():
    
    args = PARSER.parse_args()
     # 加载初始数据集
    base_smiles = []
    with open(args.input_file, 'r') as f:
        base_smiles = [line.split()[0].strip() for line in f]
        print(len(base_smiles))
    # 加载llm生成数据
    # GPT生成
    with open(args.llm_generation_file, 'r') as f:
        base_smiles_tol = base_smiles + [line.strip() for line in f if line.strip()]  # 并且合并第二个数据集
        print(len(base_smiles_tol))
     
    
    # 合并初始种群（原种群 + 独立LLM分子）
    initial_population = list(base_smiles_tol)
    print(len(initial_population))
    # #初始种群0(只有base_smiles)
    initial_population_0 = list(base_smiles)#只有原始数据集合
    print(len(initial_population_0))
    
        #
    # 初始化评估器
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    from tdc import Oracle
    qed_evaluator = Oracle(name='qed')
    sa_evaluator = Oracle(name='sa')
    # 评估初始种群
    print('''*********************初始评估*********************''')
    initial_metrics = evaluate_population(initial_population, div_eval, nov_eval, 
                                        qed_eval, sa_eval, base_smiles_tol)
    print(f"初始种群评估结果:\n{initial_metrics}")
    # 变异参数配置
    vars = {
        'rxn_library': 'all_rxns',
        'rxn_library_file': os.path.join(PROJECT_ROOT, 'autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json'),
        'function_group_library': os.path.join(PROJECT_ROOT, 'autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json'),
        'complementary_mol_directory': os.path.join(PROJECT_ROOT, 'autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir'),
        'filter_object_dict': {
            # 可根据需要添加过滤器
            # 'SA_filter': SAScoreFilter(max_score=4.5)
            #'Lipinski_filter': LipinskiStrictFilter(),
            # 'Ghose_filter': GhoseFilter(),
            # 'VandeWaterbeemd_filter': VandeWaterbeemdFilter(),
            # 添加基础结构检查
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
            
            # 尝试最多2次变异
            success = False
            for attempt in range(args.max_mutations):
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

  
    new_population = initial_population + mutation_results
    #把变异产生的新分子群存放文件中
    mutation_output_file = os.path.join(PROJECT_ROOT, "output/generation_0_mutation_new.smi")
    with open(mutation_output_file, 'w') as f:
        for smi in mutation_results:
            f.write(f"{smi}\n")
    

    # 评估新种群
    print('''*********************评估新种群*********************''')

    final_metrics = evaluate_population(new_population, div_eval, nov_eval,
                                      qed_eval, sa_eval, initial_population)
    print(f"\n变异后整个种群评估结果:\n{final_metrics}")
    #评估变异之后产生的新分子群性质
    mutation_metrics = evaluate_population(mutation_results, div_eval, nov_eval,
                                          qed_eval, sa_eval, initial_population)
    print(f"\n变异产生的新种群评估结果:\n{mutation_metrics}")

    # 保存结果
    with open(args.output_file, 'w') as f:
        for smi in new_population:
            f.write(f"{smi}\n")

if __name__ == "__main__":
    main()


