import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import random
import json
import argparse
import logging
import autogrow.operators.crossover.smiles_merge.smiles_merge as smiles_merge 
import autogrow.operators.crossover.execute_crossover as execute_crossover
import autogrow.operators.filter.execute_filters as Filter

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("crossover_finetune")
def main():
    parser = argparse.ArgumentParser(description='改进的GA交叉参数')
    parser.add_argument("--smiles_file", "-s", type=str, required=True,
                      help="输入SMILES文件路径")
    parser.add_argument("--output_file", "-o", type=str, 
                      default=os.path.join(PROJECT_ROOT, "output/generation_crossover_0.smi"),
                      help="输出文件路径")
    parser.add_argument('--config_file', type=str, default='fragevo/config_example.json', 
                      help='配置文件路径')
    parser.add_argument('--lineage_file', type=str, default=None,
                      help='可选的血统记录输出文件(JSONL)')
    parser.add_argument('--seed', type=int, default=None,
                      help='随机种子（用于保证可复现性）')
    args = parser.parse_args()    
    # 加载配置
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    seed_value = args.seed if args.seed is not None else config.get("workflow", {}).get("seed", 42)
    try:
        seed_value = int(seed_value)
    except (TypeError, ValueError):
        seed_value = 42
    random.seed(seed_value)
    enable_lineage_tracking = bool(config.get("workflow", {}).get("enable_lineage_tracking", False))
    crossover_config = config['crossover_finetune']
    # 设置日志
    logger = setup_logging()
    logger.info("开始交叉操作")    
    # 加载SMILES文件
    all_smiles = []
    with open(args.smiles_file, 'r') as f:
        all_smiles = [line.split()[0].strip() for line in f if line.strip()]
        logger.info(f"加载分子数量: {len(all_smiles)}")    
    initial_population = sorted(set(all_smiles))
    input_smiles_set = set(initial_population)
    # 构造 autogrow 风格的 (smiles, id) 对，便于“覆盖式”抽样
    ligands_list = [[smi, f"ligand_{i}"] for i, smi in enumerate(initial_population)]
    # 从配置中读取交叉参数
    number_of_crossovers = crossover_config.get(
        "number_of_crossovers",
        crossover_config.get("crossover_attempts", 20),  # backward-compat
    )
    vars = {
        'min_atom_match_mcs': crossover_config.get('min_atom_match_mcs', 4),
        'max_time_mcs_prescreen': crossover_config.get('max_time_mcs_prescreen', 1),
        'max_time_mcs_thorough': crossover_config.get('max_time_mcs_thorough', 1),
        'protanate_step': crossover_config.get('protanate_step', True),
        'number_of_crossovers': number_of_crossovers,
        'filter_object_dict': {},  # 过滤器配置（如果需要）
    }    
    crossover_attempts = vars['number_of_crossovers']
    max_attempts_multiplier = 10
    merge_attempts = 3
    
    logger.info(f"开始交叉操作，本轮目标生成 {crossover_attempts} 个新分子")
    crossed_population = []
    crossed_population_set = set()
    lineage_records = [] if enable_lineage_tracking else None
    attempts = 0
    max_attempts = crossover_attempts * max_attempts_multiplier
    react_list = ligands_list.copy()
    random.shuffle(react_list)
    
    while len(crossed_population) < crossover_attempts and attempts < max_attempts:
        attempts += 1
        try:
            # 按 autogrow4.0 的思路：优先让每个父代都有机会作为 ligand1 参与
            if not react_list:
                react_list = ligands_list.copy()
                random.shuffle(react_list)
            parent1_pair = react_list.pop()
            parent1 = parent1_pair[0]

            # 为 parent1 找一个能通过 MCS 预筛的 parent2（会遍历随机顺序，覆盖性更强）
            parent2_pair = execute_crossover.find_random_lig2(vars, ligands_list, parent1_pair)
            if not parent2_pair:
                continue
            parent2 = parent2_pair[0]

            ligand_new_smiles = None
            for _ in range(merge_attempts):
                ligand_new_smiles = smiles_merge.run_main_smiles_merge(vars, parent1, parent2)
                if ligand_new_smiles is not None:
                    break

            if ligand_new_smiles is None:
                continue

            # 去重1：交叉产物不能回到输入池(父代/GPT池)，否则浪费评估预算
            if ligand_new_smiles in input_smiles_set:
                continue
            # 去重2：交叉产物不能与本轮已生成交叉产物重复
            if ligand_new_smiles in crossed_population_set:
                continue

            if Filter.run_filter_on_just_smiles(ligand_new_smiles, vars['filter_object_dict']):
                crossed_population.append(ligand_new_smiles)
                crossed_population_set.add(ligand_new_smiles)
                if lineage_records is not None:
                    lineage_records.append({
                        "child": ligand_new_smiles,
                        "operation": "crossover",
                        "parents": [parent1, parent2]
                    })

        except Exception as e:
            logger.warning(f"交叉操作出错: {str(e)}")
            continue

    if attempts >= max_attempts:
        logger.warning(f"达到最大尝试次数 {max_attempts}，但只生成了 {len(crossed_population)} 个有效分子")
        
    logger.info(f"本轮交叉实际生成 {len(crossed_population)} 个新分子，尝试次数: {attempts}")
    
    # 保存最终结果 (只保存新交叉生成的分子)
    with open(args.output_file, 'w') as f:
        for smi in crossed_population:
            f.write(f"{smi}\n")
    logger.info(f"最终结果已保存至: {args.output_file} (仅包含新生成的分子)")

    if args.lineage_file and enable_lineage_tracking and lineage_records is not None:
        with open(args.lineage_file, 'w', encoding='utf-8') as lineage_f:
            for record in lineage_records:
                lineage_f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"血统记录已写入: {args.lineage_file}")

if __name__ == "__main__":
    main() 
