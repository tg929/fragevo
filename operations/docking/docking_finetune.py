#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接工作流程模块 (finetune)
================
提供分子对接的核心功能，接受配置字典作为参数
移除了参数解析和main函数,作为模块被主流程调用
"""
import os
import sys
import glob
import subprocess
import time
import re
import shutil
import random
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import logging
from typing import Dict, Optional, Tuple, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的模块
from autogrow.operators.convert_files.conversion_to_3d import convert_to_3d
from autogrow.operators.convert_files.conversion_to_3d import convert_sdf_to_pdbs
from autogrow.docking.docking_class.docking_file_conversion.convert_with_mgltools import MGLToolsConversion
from autogrow.docking.docking_class.docking_file_conversion.convert_with_obabel import ObabelConversion
from autogrow.operators.convert_files.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer

from autogrow.docking.docking_class.docking_class_children.vina_docking import VinaDocking
from autogrow.docking.docking_class.docking_class_children.quick_vina_2_docking import QuickVina2Docking

from autogrow.docking.execute_docking import run_docking_common
from autogrow.docking.docking_class.parent_pdbqt_converter import ParentPDBQTConverter
from autogrow.docking.docking_class.parent_dock_class import ParentDocking
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入CPU检测工具
from utils.cpu_utils import get_available_cpu_cores

def vina_dock_single(ligand_file, receptor_pdbqt, results_dir, vars):
    """ 单个分子的对接函数，静默忽略失败分子。"""    
    out_file = os.path.join(results_dir, os.path.basename(ligand_file).replace(".pdbqt", "_out.pdbqt"))
    log_file = os.path.join(results_dir, os.path.basename(ligand_file).replace(".pdbqt", ".log"))    
    cmd = [
        vars["docking_executable"],
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_file,
        "--center_x", str(vars["center_x"]),
        "--center_y", str(vars["center_y"]),
        "--center_z", str(vars["center_z"]),
        "--size_x", str(vars["size_x"]),
        "--size_y", str(vars["size_y"]),
        "--size_z", str(vars["size_z"]),
        "--out", out_file,
        "--log", log_file
    ]
    seed_value = vars.get("seed")
    if seed_value is not None:
        cmd.extend(["--seed", str(seed_value)])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    score = extract_vina_score_from_pdbqt(out_file)
    if os.path.exists(log_file):
        os.remove(log_file)
    return ligand_file, True, score

def extract_vina_score_from_pdbqt(pdbqt_file):
    """
    从输出中提取对接分数，兼容 Vina 与 QuickVina2 多种格式。
    优先解析 out.pdbqt 中含有 REMARK 的结果行；若失败，回退解析 .log。
    返回字符串形式分数（如 "-7.6"），失败返回 "NA"。
    """
    import re
    # 1) 尝试从 pdbqt 读取
    try:
        with open(pdbqt_file, "r", errors='ignore') as f:
            for line in f:
                u = line.upper()
                if "REMARK" in u and ("RESULT" in u or "VINA" in u or "QVINA" in u):
                    # 抓取行中的所有浮点数
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                    # 选择第一个带负号或第一个数字作为能量值
                    if nums:
                        # 优先选择负值
                        negs = [x for x in nums if x.startswith('-')]
                        val = negs[0] if negs else nums[0]
                        try:
                            float(val)
                            return val
                        except Exception:
                            pass
    except Exception:
        pass
    # 2) 回退到 log 文件
    log_file = pdbqt_file.replace("_out.pdbqt", ".log")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', errors='ignore') as f:
                content = f.read()
            # 常见模式：Vina / QVina2 的表格输出（第 1 行即最优构象）
            #   1         -8.3      0.000      0.000
            m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", content, flags=re.MULTILINE)
            if m:
                return m.group(1)

            # 兜底：抓取第一个“带小数点”的负数，避免误匹配引用信息中的“455-461”→“-461”
            neg_floats = re.findall(r"-\d+\.\d+", content)
            if neg_floats:
                return neg_floats[0]
        except Exception:
            pass
    return "NA"

def keep_best_docking_results(results_dir):
    """    对每个分子,只保留对接分数最好的那个构象体的结果文件(_out.pdbqt 和 .log)，其余全部删除。
    """
    print(f"正在从 {results_dir} 清理非最佳构象体的对接结果...")    
    # 1. 按分子基础ID对文件进行分组
    output_files = glob.glob(os.path.join(results_dir, "*_out.pdbqt"))#
    files_by_id = defaultdict(list)    
    for file_path in output_files:
        base_name = os.path.basename(file_path)
        # 从 'naphthalene_1__1_out.pdbqt' 中提取 'naphthalene_1'
        mol_id = base_name.split('__')[0]
        files_by_id[mol_id].append(file_path)
    # 2. 遍历每个分组，找到分数最好的文件并删除其他文件
    for mol_id, file_group in files_by_id.items():
        if len(file_group) <= 1:
            continue

        best_file = None
        best_score = float('inf') 

        # 2a. 在组内找到最佳文件
        for file_path in file_group:
            score_str = extract_vina_score_from_pdbqt(file_path)
            if score_str != "NA":
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                if score < best_score:
                    best_score = score
                    best_file = file_path

        # 2b. 删除组内非最佳的所有结果文件
        for file_path in file_group:
            if file_path != best_file:
                try:
                    # 删除 PDBQT 输出文件
                    os.remove(file_path)
                    
                    # 删除对应的 log 文件
                    log_file = file_path.replace("_out.pdbqt", ".log")
                    if os.path.exists(log_file):
                        os.remove(log_file)
                except OSError as e:
                    print(f"删除文件失败: {file_path}, 错误: {e}")

def output_smiles_scores(smiles_file, scores_dict, output_file):
    """将成功对接的结果写入文件,按对接分数排序（分数越低越好排在前面）,不包含头,不记录NA值。"""
    from rdkit import Chem
    def _normalize_smiles_remove_explicit_h(smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return smiles

    # 读取SMILES与分子名映射
    smiles_map = {}
    with open(smiles_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles_map[parts[1]] = parts[0]
    
    # 收集有效的分子和分数
    valid_molecules = []
    for mol_name, smiles in smiles_map.items():
        score = scores_dict.get(mol_name, "NA")
        # 仅收集得分有效的分子
        if score != "NA":
            norm_smiles = _normalize_smiles_remove_explicit_h(smiles)
            valid_molecules.append((norm_smiles, float(score)))
    
    # 按对接分数排序（分数越低越好，升序排列）
    valid_molecules.sort(key=lambda x: x[1])
    
    # 写入排序后的结果
    with open(output_file, "w") as out:
        for smiles, score in valid_molecules:
            out.write(f"{smiles}\t{score}\n")
    
    print(f"已将 {len(valid_molecules)} 个有效分子按对接分数排序写入 {output_file}")

class DockingWorkflow:
    """分子对接工作流程类"""    
    def __init__(self, config: Dict, generation_dir: str, ligands_file: str):
        """
        初始化工作流程        
        参数:
        :param dict config: 完整的配置参数字典
        :param str generation_dir: 当前代数专用的输出目录
        :param str ligands_file: 输入的SMILES文件路径
        """
        self.generation_dir = Path(generation_dir)
        
        # 首先从配置中初始化self.vars
        self._initialize_vars_from_config(config)
        
        # 设置输入的配体文件
        self.vars['ligands'] = ligands_file
        
        # 使用代际目录覆盖配置文件中的输出路径
        self._setup_generation_dirs()
        
        # 然后设置受体配置
        self._setup_receptor_config(config)        
        
        # 初始化运行时对象
        num_processors = self.vars.get('number_of_processors', None)
        self.vars['parallelizer'] = Parallelizer('multiprocessing', num_processors, True)
        
    def _initialize_vars_from_config(self, config: Dict):
        """从配置文件初始化vars字典"""
        # 获取对接配置部分
        docking_config = config.get('docking', {})
        performance_config = config.get('performance', {})

        def resolve_path(key):
            """如果路径是相对路径,则解析为基于项目根目录的绝对路径"""
            path_str = docking_config.get(key)
            if path_str and not os.path.isabs(path_str):
                return os.path.join(PROJECT_ROOT, path_str)
            return path_str
        
        # 优先使用性能配置中的处理器数量，如果没有则使用对接配置中的
        processor_count = performance_config.get('number_of_processors') or docking_config.get('number_of_processors')
        
        # 初始化vars字典
        self.vars = {
            # 对接配置
            'dock_choice': docking_config.get('dock_choice', 'VinaDocking'),
            'conversion_choice': docking_config.get('conversion_choice', 'MGLToolsConversion'),
            'docking_exhaustiveness': docking_config.get('docking_exhaustiveness', 8),
            'docking_num_modes': docking_config.get('docking_num_modes', 9),
            'number_of_processors': processor_count,
            'seed': docking_config.get('seed'),
            'max_variants_per_compound': docking_config.get('max_variants_per_compound', 3),
            'gypsum_thoroughness': docking_config.get('gypsum_thoroughness', 3),
            'gypsum_timeout_limit': docking_config.get('gypsum_timeout_limit', 15),
            'min_ph': docking_config.get('min_ph', 6.4),
            'max_ph': docking_config.get('max_ph', 8.4),
            'pka_precision': docking_config.get('pka_precision', 1.0),
            'debug_mode': docking_config.get('debug_mode', False),
            
            # MGLTools和对接程序路径(从对接配置块读取并解析)
            'mgltools_directory': resolve_path('mgltools_dir'),
            'mgl_python': resolve_path('mgl_python'),
            'prepare_receptor4.py': resolve_path('prepare_receptor4.py'),
            'prepare_ligand4.py': resolve_path('prepare_ligand4.py'),
            'docking_executable': resolve_path('docking_executable'),
            'timeout_vs_gtimeout': docking_config.get('timeout_vs_gtimeout', 'timeout'),
        }
        
    def _setup_generation_dirs(self):
        """创建当前代数所需的所有输出子目录"""
        self.generation_dir.mkdir(exist_ok=True, parents=True)
        
        self.vars['output_directory'] = str(self.generation_dir)
        self.vars['ligand_dir'] = str(self.generation_dir / "ligands")
        self.vars['sdf_dir'] = str(self.generation_dir / "ligands3D_SDFs")
        self.vars['pdb_dir'] = str(self.generation_dir / "ligands3D_PDBs")
        self.vars['docking_results_dir'] = str(self.generation_dir / "docking_results")
        
        # 创建这些目录
        Path(self.vars['ligand_dir']).mkdir(exist_ok=True)
        Path(self.vars['sdf_dir']).mkdir(exist_ok=True)
        Path(self.vars['pdb_dir']).mkdir(exist_ok=True)
        Path(self.vars['docking_results_dir']).mkdir(exist_ok=True)

    def _setup_receptor_config(self, config: Dict):
        """
        设置受体配置: 直接加载默认受体        
        参数:
        :param dict config: 完整的配置字典
        """       
        receptors_config = config.get('receptors', {})
        receptor_info = receptors_config.get('default_receptor')
        
        if not receptor_info or not isinstance(receptor_info, dict):
            raise ValueError("配置文件中未找到有效的 'default_receptor' 对象。")
            
        print(f"使用默认受体: {receptor_info.get('name', 'N/A')} ({receptor_info.get('description', 'No description')})")
        
        # 设置受体相关参数
        # 移除对 'paths' 的依赖,直接使用 PROJECT_ROOT
        self.vars['receptor_file'] = os.path.join(PROJECT_ROOT, receptor_info['file'])
        self.vars['filename_of_receptor'] = self.vars['receptor_file'] # 别名，用于autogrow兼容性
        self.vars['center_x'] = receptor_info['center_x']
        self.vars['center_y'] = receptor_info['center_y'] 
        self.vars['center_z'] = receptor_info['center_z']
        self.vars['size_x'] = receptor_info['size_x']
        self.vars['size_y'] = receptor_info['size_y']
        self.vars['size_z'] = receptor_info['size_z']           
        
    
    def set_receptor(self, receptor_name: str, config: Dict):
        """
        从目标列表中动态切换到指定的受体        
        参数:
        :param str receptor_name: 目标受体名称
        :param dict config: 完整的配置字典
        """         
        receptors_config = config.get('receptors', {})
        target_list = receptors_config.get('target_list', {})
        
        if receptor_name not in target_list:
            available_receptors = list(target_list.keys())
            raise ValueError(f"目标受体 '{receptor_name}' 未在 target_list 中找到。可用目标: {available_receptors}")
            
        receptor_info = target_list[receptor_name]
        
        # 更新受体相关参数, 移除对 'paths' 的依赖,直接使用 PROJECT_ROOT
        self.vars['receptor_file'] = os.path.join(PROJECT_ROOT, receptor_info['file'])
        self.vars['filename_of_receptor'] = self.vars['receptor_file']  # 兼容性
        self.vars['center_x'] = receptor_info['center_x']
        self.vars['center_y'] = receptor_info['center_y']
        self.vars['center_z'] = receptor_info['center_z']
        self.vars['size_x'] = receptor_info['size_x']
        self.vars['size_y'] = receptor_info['size_y']
        self.vars['size_z'] = receptor_info['size_z']        
        print(f"已切换到受体: {receptor_info['name']} ({receptor_info.get('description', 'No description')})")
    
    def get_target_receptors(self, config: Dict) -> List[str]:
        """
        获取所有可用于迭代的目标受体列表        
        参数:
        :param dict config: 完整的配置字典        
        返回:
        :returns: List[str]: 可用目标受体名称列表
        """                   
        receptors_config = config.get('receptors', {})
        target_list = receptors_config.get('target_list', {})
        return list(target_list.keys())
    def prepare_receptor(self) -> str:
        """
        准备受体蛋白文件。
        兼容两种配置：给出 .pdb（则使用同名 .pdbqt），或直接给出 .pdbqt。
        优先返回实际存在的 .pdbqt 文件，否则抛出异常。
        返回: str: 受体PDBQT文件路径。
        """
        print("正在验证受体文件...")
        base_path = self.vars["receptor_file"]
        candidates: List[str] = []
        lower_path = base_path.lower()
        if lower_path.endswith('.pdbqt'):
            candidates.append(base_path)
        else:
            if lower_path.endswith('.pdb'):
                candidates.append(base_path[:-4] + '.pdbqt')
            candidates.append(base_path + 'qt')
        # 去重保序
        seen = set()
        uniq = []
        for p in candidates:
            if p not in seen:
                uniq.append(p); seen.add(p)
        for p in uniq:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                print(f"成功找到受体文件: {p}")
                return p
        raise FileNotFoundError(f"未找到有效的受体PDBQT文件,尝试过: {uniq}")

    def prepare_ligands(self, smi_file: str) -> str:
        """准备配体分子，静默忽略转换失败的分子。"""
        print("准备配体分子...")
        # 1. SMILES转3D SDF
        ligand_dir = self.vars["ligand_dir"]
        if not os.path.exists(ligand_dir):
            os.makedirs(ligand_dir)
        # 1. SMILES转3D SDF
        convert_to_3d(self.vars, smi_file, ligand_dir)
        # 2. SDF转PDB
        sdf_dir = self.vars["sdf_dir"]
        convert_sdf_to_pdbs(self.vars, sdf_dir, sdf_dir)
        pdb_dir = sdf_dir + "_PDB"
        if not os.path.exists(pdb_dir):
            raise RuntimeError(f"PDB目录未生成: {pdb_dir}")
        # 3. PDB转PDBQT 
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))        
        # 实例化转换器一次，而不是在循环中
        file_conversion_class = self.pick_conversion_class(self.vars["conversion_choice"])
        file_converter = file_conversion_class(self.vars, None, test_boot=False)        
        for pdb_file in tqdm(pdb_files, desc="PDB->PDBQT转换进度"):
            pdbqt_file = pdb_file + "qt"
            # 如果文件已存在且有效，则跳过
            if os.path.exists(pdbqt_file) and os.path.getsize(pdbqt_file) > 0:
                continue            
            success, smile_name = file_converter.convert_ligand_pdb_file_to_pdbqt(pdb_file)
        return pdb_dir  # 返回包含PDBQT文件的目录

    def run_docking(self, receptor_pdbqt: str, ligand_dir: str) -> str:
        print("执行分子对接...")
        # 输出目录/路径
        results_dir = os.path.join(self.vars["output_directory"], "docking_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 配体文件查找
        ligand_files = sorted(glob.glob(os.path.join(ligand_dir, "*.pdbqt")))
        
        # 智能CPU核心数处理
        num_workers = self.vars.get("number_of_processors")
        if num_workers is None or num_workers == -1:
            # 自动检测模式：使用实时CPU检测
            available_cores, cpu_usage = get_available_cpu_cores()
            num_workers = available_cores
            print(f"自动检测到 {available_cores} 个空闲CPU核心（当前系统使用率: {cpu_usage:.1f}%）")
        else:
            num_workers = int(num_workers)
        
        # 确保至少使用1个核心
        num_workers = max(1, num_workers)
        print(f"将使用 {num_workers} 个CPU核心进行并行对接...")
        future_to_ligand = {}  # 对接任务与配体文件的映射
        scores = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for ligand_file in ligand_files:  # 遍历所有配体
                future = executor.submit(
                    vina_dock_single, ligand_file, receptor_pdbqt, results_dir, self.vars
                )
                future_to_ligand[future] = ligand_file
            
            for future in tqdm(as_completed(future_to_ligand), total=len(future_to_ligand), desc="对接进度"):
                ligand_file = future_to_ligand[future]
                try:
                    _, success, score_str = future.result()
                except Exception as exc:
                    logger.error(f"配体 {ligand_file} 对接失败: {exc}")
                    success = False
                    score_str = "NA"
                
                base_name = os.path.basename(ligand_file).replace(".pdbqt", "")
                mol_name = base_name.split('__')[0]
                if success and score_str != "NA":
                    try:
                        score_float = float(score_str)
                        current_best_score = scores.get(mol_name)
                        if current_best_score is None or score_float < float(current_best_score):
                            scores[mol_name] = score_str
                    except (ValueError, TypeError):
                        if mol_name not in scores:
                            scores[mol_name] = "NA"
                else:
                    if mol_name not in scores:
                        scores[mol_name] = "NA"
        
        # 仅保留最佳对接的mol/姿势
        keep_best_docking_results(results_dir)

        # 写出分数字典
        final_scores_file = os.path.join(results_dir, "final_scored.smi")
        output_smiles_scores(self.vars["ligands"], scores, final_scores_file)
        return final_scores_file

    @staticmethod
    def pick_conversion_class(conversion_choice: str) -> type:
        """
        选择文件转换类        
        参数:
        :param str conversion_choice: 转换方式选择        
        返回:
        :returns: type: 转换类
        """
        conversion_classes = {
            "MGLToolsConversion": MGLToolsConversion,
            "ObabelConversion": ObabelConversion
        }
        return conversion_classes.get(conversion_choice)        
    @staticmethod
    def pick_docking_class(dock_choice: str) -> type:
        """
        选择对接类        
        参数:
        :param str dock_choice: 对接程序选择
        返回:
        :returns: type: 对接类
        """
        docking_classes = {
            "VinaDocking": VinaDocking,
            "QuickVina2Docking": QuickVina2Docking
        }
        return docking_classes.get(dock_choice)

def run_molecular_docking(config: Dict, ligands_file: str, generation_dir: str, receptor_name: Optional[str] = None) -> Optional[str]:
    """
    执行完整的分子对接流程。
    
    参数:
    :param dict config: 配置字典
    :param str ligands_file: 包含SMILES和ID的文件路径
    :param str generation_dir: 当前代际的输出目录
    :param str receptor_name: (可选) 要使用的受体名称，覆盖默认受体
    
    返回:
    :return: str or None: 成功则返回包含对接分数的最终文件路径,否则返回None
    """
    logger.info("启动分子对接工作流程...")

    docking_config = config.setdefault("docking", {})
    seed_value = docking_config.get("seed")
    if seed_value is None:
        seed_value = config.get("workflow", {}).get("seed", 42)
        docking_config["seed"] = seed_value
    try:
        seed_value = int(seed_value)
    except (TypeError, ValueError):
        seed_value = 42
        docking_config["seed"] = seed_value
    random.seed(seed_value)
    
    # 实例化工作流，并传入配体文件路径
    workflow = DockingWorkflow(config, generation_dir, ligands_file)

    # 如果指定了受体，则切换到该受体
    if receptor_name:
        workflow.set_receptor(receptor_name, config)

        # 1. 准备受体
        receptor_pdbqt = workflow.prepare_receptor()
        if not receptor_pdbqt:
            logger.error("受体准备失败")
            return None
        
        # 2. 准备配体 (SMILES -> 3D PDB -> PDBQT)
        pdbqt_dir = workflow.prepare_ligands(ligands_file)
        if not pdbqt_dir:
            logger.error("配体准备失败")
            return None
            
        # 3. 执行对接
        final_results_file = workflow.run_docking(receptor_pdbqt, pdbqt_dir)
        
        if final_results_file and os.path.exists(final_results_file):
            logger.info(f"对接成功完成。最终结果保存在: {final_results_file}")
            return final_results_file
        else:
            logger.error("对接流程失败，未生成有效的输出文件。")
            return None

def main():
    """主函数，用于独立运行此脚本。"""
    import argparse
    import json
    import shutil

    parser = argparse.ArgumentParser(description="docking")
    parser.add_argument('--smiles_file', type=str, required=True, help='--smiles_file')
    parser.add_argument('--output_file', type=str, required=True, help='--output_file')
    parser.add_argument('--config_file', type=str, default='fragevo/config_fragevo.json', help='--config_file')
    parser.add_argument('--generation_dir', type=str, required=True, help='--generation_dir')
    parser.add_argument('--receptor', type=str, default=None, help='--receptor')
    # 新增：显式传递处理器数量
    parser.add_argument('--number_of_processors', type=int, default=None, help='--number_of_processors')
    parser.add_argument('--seed', type=int, default=None, help='--seed')

    args = parser.parse_args()
    
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"无法加载配置文件 {args.config_file}: {e}")
        exit(1)
        
    # 关键修复：如果通过命令行传递了处理器数量，则覆盖配置文件中的值
    if args.number_of_processors is not None:
        if 'performance' not in config:
            config['performance'] = {}
        config['performance']['number_of_processors'] = args.number_of_processors
        logger.info(f"通过命令行参数覆盖CPU核心数为: {args.number_of_processors}")
    if args.seed is not None:
        if 'docking' not in config:
            config['docking'] = {}
        config['docking']['seed'] = args.seed
        logger.info(f"通过命令行参数覆盖对接随机种子为: {args.seed}")

    # 将smiles_file参数和receptor参数传递给对接工作流
    final_output_file = run_molecular_docking(config, args.smiles_file, args.generation_dir, args.receptor)
    
    if final_output_file:
        print(f"对接工作流成功完成: {final_output_file}")
        # 将生成的最终结果文件复制到指定的output_file
        shutil.copy(final_output_file, args.output_file)
        logging.info(f"结果已成功复制到: {args.output_file}")
        exit(0)
    else:
        logging.error("对接流程失败，未生成有效的输出文件。")
        # 创建一个空的输出文件，以防止下游流程因FileNotFound而崩溃
        Path(args.output_file).touch()
        exit(1)

if __name__ == "__main__":
    main()
