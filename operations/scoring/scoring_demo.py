#我需要对比的的指标计算
import argparse
import os
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tdc import Oracle, Evaluator

# 使用TDC进行所有指标评估
qed_evaluator = Oracle('qed')
sa_evaluator = Oracle('sa')
diversity_evaluator = Evaluator(name='Diversity')
# 注意: TDC的Novelty评估器需要一个初始SMILES列表进行初始化
# 我们将在主函数中根据参数动态创建它

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

def calculate_sa_scores(smiles_list: list) -> list:
    """使用TDC Oracle批量计算SA分数。"""
    if not smiles_list:
        return []
    print(f"使用TDC批量计算 {len(smiles_list)} 个分子的SA分数...")
    return sa_evaluator(smiles_list)

def calculate_qed_scores(smiles_list: list) -> list:
    """使用TDC Oracle批量计算QED分数。"""
    if not smiles_list:
        return []
    print(f"使用TDC批量计算 {len(smiles_list)} 个分子的QED分数...")
    return qed_evaluator(smiles_list)

def load_smiles_from_file(filepath):   #加载smile
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            smiles = line.strip().split()[0] 
            if smiles:
                smiles_list.append(smiles)    
    return smiles_list

def load_smiles_and_scores_from_file(filepath):   #加载smile和score：对接之后输出文件（带分数）
    molecules = []
    scores = []
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles = parts[0]
                try:
                    score = float(parts[1])
                    molecules.append(smiles)
                    scores.append(score)
                    smiles_list.append(smiles)
                except ValueError:
                    print(f"Warning: Could not parse score for SMILES: {smiles}")
            elif len(parts) == 1: # If only SMILES is present, no score
                smiles_list.append(parts[0])    
    return smiles_list, molecules, scores

def get_rdkit_mols(smiles_list): #smiles-----mol
    mols = []
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols.append(mol)
            valid_smiles.append(s)
        else:
            print(f"Warning: Could not parse SMILES: {s}")
    return mols, valid_smiles

def calculate_docking_stats(scores):
    """Calculates Top-1, Top-10 mean, Top-100 mean docking scores."""    
    sorted_scores = sorted(scores) # Docking scores, lower is better
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else np.nan    #top1
    top10_scores = sorted_scores[:10]
    top10_mean = np.mean(top10_scores) if top10_scores else np.nan           #top10
    top100_scores = sorted_scores[:100]
    top100_mean = np.mean(top100_scores) if top100_scores else np.nan        #top100
    return top1_score, top10_mean, top100_mean

def calculate_novelty(current_smiles: list, initial_smiles_list: list) -> float:
    """使用TDC Evaluator计算新颖性。"""
    if not current_smiles:
        return 0.0
    # 正确用法: 直接按位置传入参数
    novelty_evaluator = Evaluator(name='Novelty')
    return novelty_evaluator(current_smiles, initial_smiles_list)

def calculate_top100_diversity(smiles_list: list) -> float:
    """使用TDC Evaluator计算Top-100的多样性。"""
    top_smiles = smiles_list[:min(100, len(smiles_list))]
    if not top_smiles:
        return 0.0
    return diversity_evaluator(top_smiles)

def print_calculation_results(results):    
    print("Calculation Results:")
    print(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generation of molecules.")
    parser.add_argument("--current_population_docked_file", type=str, required=True,
                        help="Path to the SMILES file of the current population with docking scores (SMILES score per line).")
    parser.add_argument("--initial_population_file", type=str, required=True,
                        help="Path to the SMILES file of the initial population (for novelty calculation).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file to save calculated metrics (e.g., results.txt or results.csv).")
    
    args = parser.parse_args()
    print(f"Processing population file: {args.current_population_docked_file}")
    print(f"Using initial population for novelty: {args.initial_population_file}")
    print(f"Saving results to: {args.output_file}")

    # 加载当前SMILES和对接分数
    current_smiles_list, scored_molecules_smiles, docking_scores = load_smiles_and_scores_from_file(args.current_population_docked_file)
    
    # 对接分数排序
    if scored_molecules_smiles and docking_scores:
        molecules_with_scores = sorted(zip(scored_molecules_smiles, docking_scores), key=lambda x: x[1])
        sorted_smiles = [s for s, _ in molecules_with_scores]
    else:
        sorted_smiles = current_smiles_list # 如果没有分数，则使用原始顺序
        
    # 1. Docking Score Metrics
    top1_score, top10_mean_score, top100_mean_score = calculate_docking_stats(docking_scores)
    
    # 定义用于计算所有属性指标的精英分子群体 (Top 100)
    smiles_for_scoring = sorted_smiles[:min(100, len(sorted_smiles))]
    score_description = f"Top {len(smiles_for_scoring)}"

    # 2. Novelty (基于Top 100精英种群)
    initial_smiles = load_smiles_from_file(args.initial_population_file)
    novelty = calculate_novelty(smiles_for_scoring, initial_smiles)
    
    # 3. Diversity (Top 100 molecules)
    diversity = calculate_top100_diversity(smiles_for_scoring)
    
    # 4. QED & SA Scores (for Top 100)
    qed_scores = calculate_qed_scores(smiles_for_scoring)
    sa_scores = calculate_sa_scores(smiles_for_scoring)
    
    mean_qed = np.mean(qed_scores) if qed_scores else np.nan
    mean_sa = np.mean(sa_scores) if sa_scores else np.nan

    # 安全地处理可能包含特殊字符的文件名
    population_filename = os.path.basename(args.current_population_docked_file)
    initial_population_filename = os.path.basename(args.initial_population_file)    
    # 为了避免f-string格式化问题，使用传统的字符串格式化
    results = "Metrics for Population: {}\n".format(population_filename)
    results += "--------------------------------------------------\n"
    results += "Total molecules processed: {}\n".format(len(current_smiles_list))
    results += "Valid RDKit molecules for properties: {}\n".format(len(sorted_smiles))
    results += "Molecules with docking scores: {}\n".format(len(docking_scores))
    results += "--------------------------------------------------\n"    
    # 处理浮点数格式化，注意处理NaN情况
    if np.isnan(top1_score): #top1
        results += "Docking Score - Top 1: N/A\n"
    else:
        results += "Docking Score - Top 1: {:.4f}\n".format(top1_score)
        
    if np.isnan(top10_mean_score): #top10
        results += "Docking Score - Top 10 Mean: N/A\n"
    else:
        results += "Docking Score - Top 10 Mean: {:.4f}\n".format(top10_mean_score)    

    if np.isnan(top100_mean_score): #top100
        results += "Docking Score - Top 100 Mean: N/A\n"
    else:
        results += "Docking Score - Top 100 Mean: {:.4f}\n".format(top100_mean_score)    
    results += "--------------------------------------------------\n"
    results += "Novelty (vs {}): {:.4f}\n".format(initial_population_filename, novelty)
    results += "Diversity (Top 100): {:.4f}\n".format(diversity)
    results += "--------------------------------------------------\n"    
    if np.isnan(mean_qed):
        results += "QED - {} Mean: N/A\n".format(score_description)
    else:
        results += "QED - {} Mean: {:.4f}\n".format(score_description, mean_qed)        
    if np.isnan(mean_sa):
        results += "SA Score - {} Mean: N/A\n".format(score_description)
    else:
        results += "SA Score - {} Mean: {:.4f}\n".format(score_description, mean_sa)    
    results += "--------------------------------------------------\n"    
    print_calculation_results(results)
    
    # Save results to output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(results)
        print(f"Results successfully saved to {args.output_file}")
    except IOError:
        print(f"Error: Could not write results to {args.output_file}")
if __name__ == "__main__":
    main()
