import os
import sys
import argparse
import logging
import subprocess
import time
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 定义receptor_info_list，包含所有受体的信息(与rga对比)
receptor_info_list = [
    ('4r6e', './pdb/4r6e.pdb', -70.76, 21.82, 28.33, 15.0, 15.0, 15.0),
    ('3pbl', './pdb/3pbl.pdb', 9, 22.5, 26, 15.0, 15.0, 15.0),
    ('1iep', './pdb/1iep.pdb', 15.6138918, 53.38013513, 15.454837, 15.0, 15.0, 15.0),
    ('2rgp', './pdb/2rgp.pdb', 16.29212, 34.870818, 92.0353, 15.0, 15.0, 15.0),
    ('3eml', './pdb/3eml.pdb', -9.06363, -7.1446, 55.86259999, 15.0, 15.0, 15.0),
    ('3ny8', './pdb/3ny8.pdb', 2.2488, 4.68495, 51.39820000000001, 15.0, 15.0, 15.0),
    ('4rlu', './pdb/4rlu.pdb', -0.73599, 22.75547, -31.23689, 15.0, 15.0, 15.0),
    ('4unn', './pdb/4unn.pdb', 5.684346153, 18.1917, -7.3715, 15.0, 15.0, 15.0),
    ('5mo4', './pdb/5mo4.pdb', -44.901, 20.490354, 8.48335, 15.0, 15.0, 15.0),
    ('7l11', './pdb/7l11.pdb', -21.81481, -4.21606, -27.98378, 15.0, 15.0, 15.0),
]

# 配置日志
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "docking_utils_demo.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("docking_utils_demo")

def extract_vina_score(log_file):
    """从Vina日志文件中提取对接得分"""
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip().startswith("1"):
                    parts = line.split()
                    if len(parts) > 1:
                        return float(parts[1])  # affinity
    except Exception as e:
        logging.error(f"解析Vina日志失败: {str(e)}")
    return None

def run_single_receptor_docking(input_file, output_file, receptor_info, mgltools_path, logger):
    """为单个受体运行对接"""
    target_id, receptor_file, center_x, center_y, center_z, size_x, size_y, size_z = receptor_info
    
    logger.info(f"开始对接受体 {target_id} 使用文件 {receptor_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建对接命令
    docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
    
    # 检查receptor_file是否存在
    receptor_path = os.path.join(PROJECT_ROOT, receptor_file)
    if not os.path.exists(receptor_path):
        logger.error(f"受体文件不存在: {receptor_path}")
        return None
        
    cmd = [
        "python", docking_script,
        "-i", input_file,
        "-r", receptor_path,
        "-o", output_file,
        "-m", mgltools_path,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z)
    ]
    
    try:
        logger.info(f"执行对接命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"对接失败: {process.stderr}")
            return None
            
        if os.path.exists(output_file):
            logger.info(f"对接完成，结果保存至: {output_file}")
            return output_file
        else:
            logger.error(f"对接结果文件不存在: {output_file}")
            return None
    except Exception as e:
        logger.error(f"对接过程出错: {str(e)}")
        return None

def dock_all_receptors(input_file, output_dir, targets, mgltools_path, logger):
    """对多个受体进行对接"""
    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有目标受体的信息
    receptors_to_dock = []
    for target in targets:
        found = False
        for info in receptor_info_list:
            if info[0] == target:
                receptors_to_dock.append(info)
                found = True
                break
        if not found:
            logger.warning(f"无法找到受体 {target} 的信息，跳过此受体")
    
    if not receptors_to_dock:
        logger.error("没有有效的受体可用于对接")
        return {}
    
    # 对每个受体进行对接
    results = {}
    for receptor_info in tqdm(receptors_to_dock, desc="对接进度"):
        target_id = receptor_info[0]
        output_file = os.path.join(output_dir, f"docked_{target_id}.smi")
        
        logger.info(f"开始对接受体 {target_id}")
        result = run_single_receptor_docking(input_file, output_file, receptor_info, mgltools_path, logger)
        
        if result:
            results[target_id] = result
            logger.info(f"受体 {target_id} 对接完成")
        else:
            logger.error(f"受体 {target_id} 对接失败")
    
    return results

def calculate_multi_receptor_scores(docking_results, output_file, logger):
    """计算多受体对接的综合得分，并按分数排序（分数越低越好排在前面）"""
    logger.info("计算多受体对接综合得分")    
    # 读取所有对接结果
    molecules = {}  # 分子SMILES -> {受体 -> 得分}    
    for target, result_file in docking_results.items():
        try:
            with open(result_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            smiles = parts[0]
                            score = float(parts[1])
                            
                            if smiles not in molecules:
                                molecules[smiles] = {}
                            
                            molecules[smiles][target] = score
        except Exception as e:
            logger.error(f"读取对接结果文件 {result_file} 失败: {str(e)}")
    
    # 计算每个分子的综合得分
    # 使用平均得分作为综合得分，得分越小越好
    combined_scores = {}
    for smiles, target_scores in molecules.items():
        if not target_scores:  # 如果没有任何对接结果
            continue
            
        # 计算平均得分
        avg_score = sum(target_scores.values()) / len(target_scores)
        combined_scores[smiles] = avg_score
    
    # 按得分从小到大排序（得分越小越好）
    sorted_molecules = sorted(combined_scores.items(), key=lambda x: x[1])
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        for smiles, score in sorted_molecules:
            f.write(f"{smiles}\t{score:.4f}\n")
    
    logger.info(f"多受体综合得分计算完成，已将 {len(sorted_molecules)} 个分子按综合得分排序写入文件: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='多受体autodock vina分子对接脚本')
    parser.add_argument('-i', '--input', required=True, help='输入SMILES文件')
    parser.add_argument('-o', '--output_dir', default="output_docking_utils", help='输出目录')
    parser.add_argument('--targets', nargs='+', default=['4r6e', '3pbl', '1iep', '2rgp', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11'], help='受体蛋白列表')
    parser.add_argument('-m', '--mgltools_path', default=os.path.join(PROJECT_ROOT, "mgltools_x86_64Linux2_1.5.6"), help='MGLTools安装路径')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir)
    
    # 检查MGLTools路径
    if not os.path.exists(args.mgltools_path):
        logger.error(f"MGLTools路径不存在: {args.mgltools_path}")
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    
    try:
        # 读取SMILES
        with open(args.input) as f:
            smiles_count = sum(1 for line in f if line.strip())
        logger.info(f"读取到 {smiles_count} 个分子")
        
        if smiles_count == 0:
            logger.error("输入文件中没有有效的SMILES分子")
            sys.exit(1)
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        sys.exit(1)
    
    # 创建对接结果目录
    docking_dir = os.path.join(output_dir, "docking_results")
    os.makedirs(docking_dir, exist_ok=True)
    
    # 对所有受体进行对接
    start_time = time.time()
    logger.info(f"开始对接，目标受体: {args.targets}")
    
    docking_results = dock_all_receptors(args.input, docking_dir, args.targets, args.mgltools_path, logger)
    
    if not docking_results:
        logger.error("所有受体对接均失败")
        sys.exit(1)
    
    # 计算综合得分
    combined_output = os.path.join(output_dir, "combined_docking_scores.smi")
    calculate_multi_receptor_scores(docking_results, combined_output, logger)
    
    end_time = time.time()
    logger.info(f"所有对接完成! 总耗时: {end_time - start_time:.2f}秒")
    logger.info(f"综合得分结果保存至: {combined_output}")

if __name__ == "__main__":
    main()
