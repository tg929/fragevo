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
import autogrow.operators.crossover.smiles_merge.smiles_merge as smiles_merge 
import autogrow.operators.crossover.execute_crossover as execute_crossover
import autogrow.operators.filter.execute_filters as Filter


PARSER = argparse.ArgumentParser()

PARSER = argparse.ArgumentParser(description='GA crossover parameters')
PARSER.add_argument("--source_compound_file", "-s",type=str, required=True,help="萘衍生物数据集路径")
PARSER.add_argument("--llm_generation_file", "-l",type=str, default=os.path.join(PROJECT_ROOT, "fragmlm/output/test0/crossovered0_frags_new_0.smi"))
PARSER.add_argument("--output_file", "-o",type=str, default=os.path.join(PROJECT_ROOT, "output/generation_crossover_0.smi"))
PARSER.add_argument("--crossover_rate", type=float, default=0.8)
#ARSER.add_argument("--max_crossover_attempts", type=int, default=1000)

PARSER.add_argument("--crossover_attempts", type=int, default=1)#设置交叉次数（尝试交叉次数)
# 初始化评估器
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

# 评估种群函数
def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    results = {
        'diversity': div_eval(smiles_list),
        'novelty': nov_eval(smiles_list, ref_smiles),
        'avg_qed': np.mean([qed_eval(s) for s in smiles_list]),
        'avg_sa': np.mean([sa_eval(s) for s in smiles_list]),
        'num_valid': len(smiles_list)#第一个参数：传入种群数据集
    }
    return results

# 主逻辑修改
def main():
    args = PARSER.parse_args()
    
    # 加载初始数据集
    base_smiles = []
    with open(args.source_compound_file, 'r') as f:
        base_smiles = [line.split()[0].strip() for line in f]
        print(len(base_smiles))
    # 加载GPT生成分子数据集
    with open(args.llm_generation_file, 'r') as f:
        base_smiles_tol = base_smiles + [line.strip() for line in f if line.strip()]  # 并且合并第二个数据集
        print(len(base_smiles_tol))
                      
    
    initial_population = list(base_smiles_tol)  # 合并初始种群   
    print(len(initial_population))
    # #初始种群0(只有base_smiles)
    initial_population_0 = list(base_smiles)#只有原始数据集合
    print(len(initial_population_0))
    
    
    # 初始化评估器(四个计算量)
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    print('''*****************************初始评估*******************************''')
    #评估最最初始种群（不加入llm生成）(没必要加了)
    # crossed_source_metrics = evaluate_population(initial_population_0, div_eval, nov_eval,
    #                                     qed_eval, sa_eval, base_smiles)
    # print(f"\n初始种群(无LLM_generation)分子群评估结果:\n{crossed_source_metrics}")    
    # 评估初始种(合并数据集和llm生成)
    initial_metrics = evaluate_population(initial_population, div_eval, nov_eval, 
                                        qed_eval, sa_eval, base_smiles)
    print(f"初始种群(含LLM_generation)评估结果:\n{initial_metrics}")
    print('''*****************************初始评估完成*******************************''')
    print('''*******************************开始交叉*********************************''')
    # 执行交叉操作
    crossed_population = []   #交叉后新种群
    vars = {
        #在execute_crossover.test_for_msc()中用到
    'min_atom_match_mcs': 4,
    'max_time_mcs_prescreen': 1, #MCS预筛选最大时间（秒）
    'max_time_mcs_thorough': 1,#MCS详细计算阶段最大时间（秒）

    'protanate_step': True,  #是否执行质子化步骤
    'number_of_crossovers': args.crossover_attempts,
    #'convert_glucose':False,#是否将葡萄糖转换为葡萄糖酸（线性表示）

        #在smiles_merge.run_main_smiles_merge()中用到
    'filter_object_dict': {},#过滤对象字典
    #'rxn_library': 'click_chem_rxns',  # 反应库参数
    'max_variants_per_compound': 1,    # 每个化合物最大变体数 
     
    # 'parallelizer': None,     
    # 'printers': None,
    'debug_mode': False,

    #构象生成
        #在Filter.run_filter_on_just_smiles()中用到
    'gypsum_timeout_limit': 120.0, #分子构象生成超时时间限制（秒）
    'gypsum_thoroughness': 3, #构象生成尝试次数
    #物化参数
    # 'min_ph': 6.4,  #最小pH值（用于质子化状态）
    # 'max_ph': 8.4, #最大pH值
    # 'pka_precision': 1.0    #pKa精度    
    }

    with tqdm(total=args.crossover_attempts, desc="Performing crossovers") as pbar:
        while len(crossed_population) < args.crossover_attempts:
            parent1, parent2 = random.sample(initial_population, 2)
            # ==== 新增交叉核心逻辑 ====
            # 1. 转换SMILES为分子对象
            try:
                mol1 = execute_crossover.convert_mol_from_smiles(parent1)
                mol2 = execute_crossover.convert_mol_from_smiles(parent2)
                if mol1 is None or mol2 is None:
                    continue
            except:
                continue
                # 2. 检查MCS（最大公共子结构）
                mcs_result = execute_crossover.test_for_mcs(vars, mol1, mol2)
                if mcs_result is None:
                    continue  # 没有足够大的公共结构
            
            # 3. 多次尝试合并
            ligand_new_smiles = None
            #尝试3次交叉合并，直到成功为止
            for attempt in range(3):
                ligand_new_smiles = smiles_merge.run_main_smiles_merge(vars, parent1, parent2)
                if ligand_new_smiles is not None:
                    break                    
            if ligand_new_smiles is None:
                continue
            # ==== 新增交叉核心逻辑结束 ====
            # 过滤新生成的分子 
            if Filter.run_filter_on_just_smiles(ligand_new_smiles, vars['filter_object_dict']):
                crossed_population.append(ligand_new_smiles)
                pbar.update(1)
    # #把新生成的crossed_population写进文件保存
    output_crossed_file = os.path.join(PROJECT_ROOT, "output/generation_0_crossed_new.smi")
    with open(output_crossed_file, 'w') as f:
        for smi in crossed_population:
            f.write(f"{smi}\n")
    #评估
    # 新种群
    print('''*******************************交叉完成*********************************''')
    new_population = initial_population + crossed_population  #含有LLM的初始+交叉生成的
    #评估交叉后  新种群
    #评估下一代（包含初始和新生成）
    print('''*******************************交叉后新种群评估*********************************''')
    crossed_metrics = evaluate_population(new_population, div_eval, nov_eval,
                                        qed_eval, sa_eval, base_smiles)
    print(f"\n初始种群(含LLM)交叉后新种群(聚合初始与新生成分子群)评估结果:\n{crossed_metrics}")
    

    #评估下一代（不包含初始时刻，只含有交叉之后新生成个体分子群）
    crossed_new_metrics = evaluate_population(crossed_population, div_eval, nov_eval,
                                        qed_eval, sa_eval, base_smiles)
    print(f"\n交叉后新生成分子群评估结果:\n{crossed_new_metrics}")


    # 保存结果
    with open(args.output_file, 'w') as f:
        for smi in new_population:
            f.write(f"{smi}\n")

if __name__ == "__main__":
    main()