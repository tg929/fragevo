#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接工作流程模块 (finetune)
"""
import os
import sys
import glob
import subprocess
import time
import re
import shutil
import random
import tempfile
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import logging
from typing import Dict, Optional, Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cpu_utils import get_available_cpu_cores

def _run_cmd(cmd: List[str], timeout_seconds: int) -> bool:
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + max(1, int(timeout_seconds))
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
        return False
    return proc.returncode == 0

def _normalize_smiles_remove_explicit_h(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def _obabel_fast_vina_dock_single(
    idx: int,
    smiles: str,
    receptor_pdbqt: str,
    vina_executable: str,
    obabel_path: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    cpu_per_dock: int,
    exhaustiveness: int,
    num_modes: int,
    seed: Optional[int],
    gen3d_timeout: int,
    convert_timeout: int,
    dock_timeout: int,
    temp_dir: str,
    keep_temp: bool,
) -> Tuple[int, Optional[float]]:
    pid = os.getpid()
    mol_file = os.path.join(temp_dir, f"ligand_{pid}_{idx}.mol")
    pdbqt_file = os.path.join(temp_dir, f"ligand_{pid}_{idx}.pdbqt")
    out_pdbqt = os.path.join(temp_dir, f"dock_{pid}_{idx}.pdbqt")

    ok = _run_cmd([obabel_path, f"-:{smiles}", "--gen3D", "-O", mol_file], gen3d_timeout)
    if ok:
        ok = _run_cmd([obabel_path, "-imol", mol_file, "-opdbqt", "-O", pdbqt_file], convert_timeout)

    if ok:
        cmd = [
            vina_executable,
            "--receptor", receptor_pdbqt,
            "--ligand", pdbqt_file,
            "--out", out_pdbqt,
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
            "--cpu", str(cpu_per_dock),
            "--num_modes", str(num_modes),
            "--exhaustiveness", str(exhaustiveness),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(int(seed))])
        ok = _run_cmd(cmd, dock_timeout)

    score: Optional[float] = None
    if ok and os.path.exists(out_pdbqt):
        score_str = extract_vina_score_from_pdbqt(out_pdbqt)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", str(score_str)):
            score = float(score_str)

    if not keep_temp:
        for p in (mol_file, pdbqt_file, out_pdbqt):
            if os.path.exists(p):
                os.remove(p)

    return idx, score

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

    # 关键性能修复：显式控制 docking 程序使用的 CPU 核心数，避免与外层并行发生严重超卖（oversubscription）。
    cpu_per_dock = vars.get("_resolved_cpu_per_dock") or vars.get("docking_cpu_per_dock")
    if cpu_per_dock is not None:
        cpu_str = str(cpu_per_dock).strip()
        cpu_per_dock = int(cpu_str) if re.fullmatch(r"\d+", cpu_str) else None
    if cpu_per_dock is not None and cpu_per_dock > 0:
        cmd.extend(["--cpu", str(cpu_per_dock)])

    # 可选：如果用户希望配置文件里的参数生效，则传入 --exhaustiveness/--num_modes
    if vars.get("pass_vina_params", False):
        exhaustiveness = vars.get("docking_exhaustiveness")
        num_modes = vars.get("docking_num_modes")
        if exhaustiveness is not None:
            cmd.extend(["--exhaustiveness", str(int(exhaustiveness))])
        if num_modes is not None:
            cmd.extend(["--num_modes", str(int(num_modes))])

    seed_value = vars.get("seed")
    if seed_value is not None:
        cmd.extend(["--seed", str(seed_value)])

    timeout_seconds = vars.get("docking_timeout_seconds", 300)
    timeout_str = str(timeout_seconds).strip()
    timeout_seconds = int(timeout_str) if re.fullmatch(r"\d+", timeout_str) else 300

    success = _run_cmd(cmd, timeout_seconds)
    score = extract_vina_score_from_pdbqt(out_file) if success else "NA"
    if os.path.exists(log_file):
        os.remove(log_file)
    return ligand_file, success, score

def extract_vina_score_from_pdbqt(pdbqt_file):
    """
    从输出中提取对接分数，兼容 Vina 与 QuickVina2 多种格式。
    优先解析 out.pdbqt 中含有 REMARK 的结果行
    返回字符串形式分数（如 "-7.6"），失败返回 "NA"。
    """
    if not pdbqt_file or not os.path.exists(pdbqt_file):
        return "NA"
    for line in open(pdbqt_file, "r", errors="ignore"):
        u = line.upper()
        if "REMARK" not in u or ("RESULT" not in u and "VINA" not in u and "QVINA" not in u):
            continue
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        if not nums:
            continue
        negs = [x for x in nums if x.startswith("-")]
        val = negs[0] if negs else nums[0]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", val):
            return val
    
    log_file = pdbqt_file.replace("_out.pdbqt", ".log")
    if os.path.exists(log_file):
        content = open(log_file, "r", errors="ignore").read()
        m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", content, flags=re.MULTILINE)
        if m:
            return m.group(1)
        neg_floats = re.findall(r"-\d+\.\d+", content)
        if neg_floats:
            return neg_floats[0]
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
            if score_str != "NA" and re.fullmatch(r"-?\d+(?:\.\d+)?", str(score_str)):
                score = float(score_str)
                if score < best_score:
                    best_score = score
                    best_file = file_path

        # 2b. 删除组内非最佳的所有结果文件
        for file_path in file_group:
            if file_path != best_file:
                if os.path.exists(file_path):
                    os.remove(file_path)
                log_file = file_path.replace("_out.pdbqt", ".log")
                if os.path.exists(log_file):
                    os.remove(log_file)

def output_smiles_scores(smiles_file, scores_dict, output_file):
    """将成功对接的结果写入文件,按对接分数排序（分数越低越好排在前面）,不包含头,不记录NA值。"""
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
        if score != "NA":
            norm_smiles = _normalize_smiles_remove_explicit_h(smiles)
            valid_molecules.append((norm_smiles, float(score)))
    
    # 按对接分数排序（分数越低越好，升序排列）
    valid_molecules.sort(key=lambda x: x[1])    
   
    with open(output_file, "w") as out:
        for smiles, score in valid_molecules:
            out.write(f"{smiles}\t{score}\n")
    
    print(f"已将 {len(valid_molecules)} 个有效分子按对接分数排序写入 {output_file}")

class DockingWorkflow:
    """分子对接工作流程类"""    
    def __init__(self, config: Dict, generation_dir: str, ligands_file: str):
       
        self.generation_dir = Path(generation_dir)        
        self._initialize_vars_from_config(config)        
        self.vars['ligands'] = ligands_file  
        
        self._setup_generation_dirs()    
      
        self._setup_receptor_config(config)       
        
        num_processors = self.vars.get('number_of_processors', None)
        self.vars['parallelizer'] = Parallelizer('multiprocessing', num_processors, True)
        
    def _initialize_vars_from_config(self, config: Dict):
        """从配置文件初始化vars字典"""        
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
            'pipeline': docking_config.get('pipeline', 'autogrow'),
            'dock_choice': docking_config.get('dock_choice', 'VinaDocking'),
            'conversion_choice': docking_config.get('conversion_choice', 'MGLToolsConversion'),
            'docking_exhaustiveness': docking_config.get('docking_exhaustiveness', 8),
            'docking_num_modes': docking_config.get('docking_num_modes', 9),
            'pass_vina_params': docking_config.get('pass_vina_params', False),
            'number_of_processors': processor_count,
            'seed': docking_config.get('seed'),
            'max_variants_per_compound': docking_config.get('max_variants_per_compound', 3),
            'gypsum_thoroughness': docking_config.get('gypsum_thoroughness', 3),
            'gypsum_timeout_limit': docking_config.get('gypsum_timeout_limit', 15),
            'docking_timeout_seconds': docking_config.get('docking_timeout_seconds', 300),
            'docking_cpu_per_dock': docking_config.get('docking_cpu_per_dock'),
            # obabel 快速流程参数
            'obabel_path': resolve_path('obabel_path'),
            'obabel_gen3d_timeout_seconds': docking_config.get('obabel_gen3d_timeout_seconds', 30),
            'obabel_convert_timeout_seconds': docking_config.get('obabel_convert_timeout_seconds', 30),
            'obabel_fail_score': docking_config.get('obabel_fail_score', 99.9),
            'obabel_filter_score_above': docking_config.get('obabel_filter_score_above', 50.0),
            'obabel_keep_temp': docking_config.get('obabel_keep_temp', False),
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

        # 1) SMILES -> 3D SDF -> PDB（convert_to_3d 内部已包含 SDF->PDB 的转换）
        convert_to_3d(self.vars, smi_file, ligand_dir)

        # convert_to_3d 会把 PDB 输出到 `${ligand_dir}_PDB/`，避免重复再跑一次 convert_sdf_to_pdbs（会额外生成 `${sdf_dir}_PDB/`，浪费大量时间）
        pdb_dir = ligand_dir + "_PDB"
        if not os.path.exists(pdb_dir):
            raise RuntimeError(f"PDB目录未生成: {pdb_dir}")
        # 3. PDB转PDBQT 
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))        
        # 实例化转换器一次，而不是在循环中
        file_conversion_class = self.pick_conversion_class(self.vars["conversion_choice"])
        if file_conversion_class is ObabelConversion and not self.vars.get("obabel_path"):
            raise RuntimeError(
                "conversion_choice=ObabelConversion 但未找到 obabel 可执行文件；"
                "请在配置里设置 docking.obabel_path 或确保 `obabel` 在 PATH 中。"
            )
        file_converter = file_conversion_class(self.vars, None, test_boot=False)        
        for pdb_file in tqdm(pdb_files, desc="PDB->PDBQT转换进度"):
            pdbqt_file = pdb_file + "qt"
            # 如果文件已存在且有效，则跳过
            if os.path.exists(pdbqt_file) and os.path.getsize(pdbqt_file) > 0:
                continue            
            success, smile_name = file_converter.convert_ligand_pdb_file_to_pdbqt(pdb_file)
        return pdb_dir  # 返回包含PDBQT文件的目录

    def _resolve_parallel_settings(self) -> Tuple[int, int]:
        """
        解析并行设置，返回:
        - num_workers: 外层并行任务数
        - cpu_per_dock: 传给 vina/qvina 的 --cpu
        """
        num_workers = self.vars.get("number_of_processors")
        available_cores = None
        if num_workers is None or num_workers == -1:
            available_cores, cpu_usage = get_available_cpu_cores()
            num_workers = available_cores
            print(f"自动检测到 {available_cores} 个空闲CPU核心（当前系统使用率: {cpu_usage:.1f}%）")
        else:
            num_workers = int(num_workers)
        num_workers = max(1, int(num_workers))

        cpu_per_dock = self.vars.get("docking_cpu_per_dock")
        if cpu_per_dock is None:
            total_for_calc = available_cores if available_cores else (os.cpu_count() or num_workers)
            cpu_per_dock = max(1, int(total_for_calc) // num_workers)
        else:
            cpu_per_dock = max(1, int(cpu_per_dock))
            total_for_calc = available_cores if available_cores else (os.cpu_count() or num_workers)
            num_workers = min(num_workers, max(1, int(total_for_calc) // cpu_per_dock))
        return num_workers, cpu_per_dock

    def run_obabel_fast_docking(self, receptor_pdbqt: str, smiles_file: str) -> str:
        """
        使用 obabel 快速 pipeline（bd3lms 风格）：SMILES -> obabel gen3D -> obabel pdbqt -> vina/qvina。
        输出 `docking_results/final_scored.smi`，格式: `SMILES\\tScore`，按分数升序。
        """
        print("执行分子对接 (obabel_fast)...")

        results_dir = os.path.join(self.vars["output_directory"], "docking_results")
        os.makedirs(results_dir, exist_ok=True)

        tmp_root = os.path.join(self.vars["output_directory"], "obabel_docking_tmp")
        os.makedirs(tmp_root, exist_ok=True)

        # 读取 SMILES（输入行格式允许: "SMILES [NAME]"；只取第 1 列）
        smiles_list: List[str] = []
        with open(smiles_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                smiles_list.append(s.split()[0])

        num_workers, cpu_per_dock = self._resolve_parallel_settings()
        print(f"将使用 {num_workers} 个并行任务进行对接（每个任务 --cpu {cpu_per_dock}）...")

        # 每次运行使用一个独立临时目录，避免历史残留
        temp_dir = tempfile.mkdtemp(prefix="obabel_", dir=tmp_root)
        keep_temp = bool(self.vars.get("obabel_keep_temp", False))

        scores: Dict[str, float] = {}
        fail_score = float(self.vars.get("obabel_fail_score", 99.9))
        filter_above = float(self.vars.get("obabel_filter_score_above", 50.0))
        obabel_path = self.vars["obabel_path"]
        vina_executable = self.vars["docking_executable"]
        exhaustiveness = int(self.vars.get("docking_exhaustiveness", 1))
        num_modes = int(self.vars.get("docking_num_modes", 10))
        gen3d_timeout = int(self.vars.get("obabel_gen3d_timeout_seconds", 30))
        convert_timeout = int(self.vars.get("obabel_convert_timeout_seconds", 30))
        dock_timeout = int(self.vars.get("docking_timeout_seconds", 100))
        seed = self.vars.get("seed")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for idx, smi in enumerate(smiles_list):
                futures.append(
                    executor.submit(
                        _obabel_fast_vina_dock_single,
                        idx,
                        smi,
                        receptor_pdbqt,
                        vina_executable,
                        obabel_path,
                        float(self.vars["center_x"]),
                        float(self.vars["center_y"]),
                        float(self.vars["center_z"]),
                        float(self.vars["size_x"]),
                        float(self.vars["size_y"]),
                        float(self.vars["size_z"]),
                        int(cpu_per_dock),
                        exhaustiveness,
                        num_modes,
                        seed,
                        gen3d_timeout,
                        convert_timeout,
                        dock_timeout,
                        temp_dir,
                        keep_temp,
                    )
                )

            for fut in tqdm(as_completed(futures), total=len(futures), desc="对接进度"):
                idx, score_float = fut.result()
                if score_float is None or score_float >= fail_score or score_float > filter_above:
                    continue
                smi = _normalize_smiles_remove_explicit_h(smiles_list[idx])
                best = scores.get(smi)
                if best is None or score_float < best:
                    scores[smi] = score_float

        final_scores_file = os.path.join(results_dir, "final_scored.smi")
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        with open(final_scores_file, "w", encoding="utf-8") as out:
            for smi, sc in sorted_items:
                out.write(f"{smi}\t{sc}\n")
        print(f"已将 {len(sorted_items)} 个有效分子按对接分数排序写入 {final_scores_file}")

        if not keep_temp:
            shutil.rmtree(temp_dir)

        return final_scores_file

    def run_docking(self, receptor_pdbqt: str, ligand_dir: str) -> str:
        print("执行分子对接...")
        # 输出目录/路径
        results_dir = os.path.join(self.vars["output_directory"], "docking_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 配体文件查找
        ligand_files = sorted(glob.glob(os.path.join(ligand_dir, "*.pdbqt")))
        
        num_workers, cpu_per_dock = self._resolve_parallel_settings()
        self.vars["_resolved_cpu_per_dock"] = cpu_per_dock

        print(f"将使用 {num_workers} 个并行任务进行对接（每个任务 --cpu {cpu_per_dock}）...")
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
                _, success, score_str = future.result()
                
                base_name = os.path.basename(ligand_file).replace(".pdbqt", "")
                mol_name = base_name.split('__')[0]
                if success and score_str != "NA":
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", str(score_str)):
                        score_float = float(score_str)
                        current_best_score = scores.get(mol_name)
                        if current_best_score is None or score_float < float(current_best_score):
                            scores[mol_name] = score_str
                    elif mol_name not in scores:
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
    seed_str = str(seed_value).strip()
    seed_value = int(seed_str) if re.fullmatch(r"-?\d+", seed_str) else 42
    docking_config["seed"] = seed_value
    random.seed(seed_value)
    
    # 实例化工作流，并传入配体文件路径
    workflow = DockingWorkflow(config, generation_dir, ligands_file)

    # 如果指定了受体，则切换到该受体；否则使用 __init__ 中加载的默认受体
    if receptor_name:
        workflow.set_receptor(receptor_name, config)

    # 1. 准备受体
    receptor_pdbqt = workflow.prepare_receptor()
    if not receptor_pdbqt:
        logger.error("受体准备失败")
        return None

    pipeline = (workflow.vars.get('pipeline') or 'autogrow').strip().lower()

    # 2. 根据 pipeline 执行不同对接流程
    if pipeline in ('obabel_fast', 'obabel-fast'):
        obabel_path = workflow.vars.get('obabel_path')
        if not obabel_path:
            # Avoid cryptic NoneType crashes.
            raise RuntimeError(
                "docking.pipeline=obabel_fast 但未设置 docking.obabel_path，且系统 PATH 中也未发现 obabel。"
            )
        final_results_file = workflow.run_obabel_fast_docking(receptor_pdbqt, ligands_file)
    else:
        # Default AutoGrow-style pipeline: SMILES -> 3D -> PDBQT -> docking
        ligand_dir = workflow.prepare_ligands(ligands_file)
        final_results_file = workflow.run_docking(receptor_pdbqt, ligand_dir)

    if final_results_file and os.path.exists(final_results_file):
        logger.info(f"对接成功完成。最终结果保存在: {final_results_file}")
        return final_results_file

    logger.error("对接流程失败，未生成有效的输出文件。")
    return None

def main():
    """主函数，用于独立运行此脚本。"""
    import argparse
    import errno
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
    
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"无法加载配置文件 {args.config_file}")
    with open(args.config_file, "r") as f:
        config = json.load(f)       
    
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
    final_output_file = run_molecular_docking(config, args.smiles_file, args.generation_dir, args.receptor)    
    if final_output_file:
        print(f"对接工作流成功完成: {final_output_file}")
        # 将生成的最终结果文件写入到指定的 output_file。
        # 当磁盘空间不足时，copy 可能失败；此时尝试用 os.replace 直接移动文件（同盘 rename），避免额外空间占用。
        try:
            shutil.copy(final_output_file, args.output_file)
            logging.info(f"结果已成功复制到: {args.output_file}")
            exit(0)
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                try:
                    os.replace(final_output_file, args.output_file)
                    logging.warning(
                        f"磁盘空间不足，已将结果文件从 {final_output_file} 直接移动为 {args.output_file}（未保留原文件）。"
                    )
                    exit(0)
                except Exception:
                    raise
            raise
    else:
        logging.error("对接流程失败，未生成有效的输出文件。")        
        Path(args.output_file).touch()
        exit(1)

if __name__ == "__main__":
    main()
