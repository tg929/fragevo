import os
import sys
import glob
import subprocess
import time
import re
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
# 导入必要的模块
from autogrow.operators.convert_files.conversion_to_3d import convert_to_3d
from autogrow.operators.convert_files.conversion_to_3d import convert_sdf_to_pdbs
from autogrow.docking.docking_class.docking_class_children.vina_docking import VinaDocking
from autogrow.docking.docking_class.docking_file_conversion.convert_with_mgltools import MGLToolsConversion
def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "docking.log")),
            logging.StreamHandler()
        ]
    )

def extract_vina_score(pdbqt_file):    
    """
    (已修复) 从PDBQT输出文件中提取Vina对接分数 (最佳模式)
    """
    try:
        with open(pdbqt_file, "r") as f:
            for line in f:
                # Vina在PDBQT文件的开头用这种格式存储结果
                if "REMARK VINA RESULT:" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        # 分数是第4个部分 (索引为3)
                        return float(parts[3])
    except Exception as e:
        logging.error(f"从PDBQT文件 {pdbqt_file} 解析分数失败: {e}")
    return None

def keep_one_pdb_per_smiles(pdb_dir):
    """
    只保留每个SMILES编号的第一个PDB文件,其余全部删除
    """
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    seen = set()
    pattern = re.compile(r"(naphthalene_\d+)")
    for pdb_file in pdb_files:
        base = os.path.basename(pdb_file)
        # 提取SMILES编号（如naphthalene_1）
        m = pattern.search(base)
        if m:
            key = m.group(1)
            if key in seen:
                os.remove(pdb_file)
            else:
                seen.add(key)

def vina_dock_single(ligand_file, receptor_pdbqt, results_dir, vars):
    """单个分子的对接函数，用于并行处理"""
    out_file = os.path.join(
        results_dir,
        os.path.basename(ligand_file).replace(".pdbqt", "_out.pdbqt")
    )
    log_file = os.path.join(
        results_dir,
        os.path.basename(ligand_file).replace(".pdbqt", ".log")
    )
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
        "--log", log_file,
        "--cpu", "1"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 关键修复: 从 out_file (PDBQT) 而不是 log_file 读取分数
        score = extract_vina_score(out_file)
        # 删除log文件
        if os.path.exists(log_file):
            os.remove(log_file)
        return ligand_file, True, score
    except Exception as e:
        logging.error(f"对接失败: {ligand_file}, 错误: {str(e)}")
        return ligand_file, False, None

# 对接执行器类
class DockingExecutor:
    def __init__(self, receptor_pdb, output_dir, mgltools_path, number_of_processors,
                 center_x, center_y, center_z, size_x, size_y, size_z):
        self.receptor_pdb = receptor_pdb
        self.output_dir = os.path.abspath(output_dir)
        self.mgltools_path = mgltools_path
        self.number_of_processors = number_of_processors
        self._validate_paths()
        
        # 创建必要的目录
        self.ligand_dir = os.path.join(self.output_dir, "ligands")
        self.sdf_dir = os.path.join(self.output_dir, "ligands3D_SDFs")
        self.pdb_dir = os.path.join(self.output_dir, "ligands3D_PDBs")
        self.results_dir = os.path.join(self.output_dir, "docking_results")
        
        for dir_path in [self.ligand_dir, self.sdf_dir, self.pdb_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化对接参数，使用传入的对接盒子信息
        self.docking_params = {
            'center_x': center_x,
            'center_y': center_y,
            'center_z': center_z,
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'exhaustiveness': 8,
            'num_modes': 9,
            'timeout': 120
        }        
        # 准备VINA需要的变量
        self.vars = self._prepare_vars()        
        # 初始化文件转换器
        self.converter = MGLToolsConversion(
            vars=self.vars, 
            receptor_file=receptor_pdb,
            test_boot=False
        )
        
        # 初始化对接器
        self.docker = VinaDocking(
            vars=self.vars,
            receptor_file=receptor_pdb,
            file_conversion_class_object=self.converter,
            test_boot=False
        )

    def _prepare_vars(self):
        """准备和Autogrow一样的所需的参数字典"""
        return {
            'filename_of_receptor': self.receptor_pdb,
            'mgl_python': os.path.join(self.mgltools_path, "bin/pythonsh"),
            'prepare_receptor4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
            'prepare_ligand4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"), 
            'docking_executable': os.path.join(PROJECT_ROOT, "autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina"),
            'number_of_processors': self.number_of_processors,
            'debug_mode': False,
            'timeout_vs_gtimeout': 'timeout',  
            'docking_timeout_limit': self.docking_params['timeout'],
            'center_x': self.docking_params['center_x'],
            'center_y': self.docking_params['center_y'],
            'center_z': self.docking_params['center_z'],
            'size_x': self.docking_params['size_x'],
            'size_y': self.docking_params['size_y'],
            'size_z': self.docking_params['size_z'],
            'docking_exhaustiveness': self.docking_params['exhaustiveness'],
            'docking_num_modes': self.docking_params['num_modes'],
            'environment': {                   
                'MGLPY': os.path.join(self.mgltools_path, "bin/python"),
                'PYTHONPATH': f"{os.path.join(self.mgltools_path, 'MGLToolsPckgs')}:{os.environ.get('PYTHONPATH', '')}"
            },
            'output_directory': self.output_dir,
            'sdf_dir': self.sdf_dir,
            'pdb_dir': self.pdb_dir,
            'ligand_dir': self.ligand_dir
        }

    def _validate_paths(self):
        """验证必要路径"""
        required_files = {
            'prepare_receptor4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
            'prepare_ligand4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"),
            'pythonsh': os.path.join(self.mgltools_path, "bin/pythonsh")
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file missing: {name} -> {path}")

    def generate_3d_conformer(self, mol, max_attempts=5):
        """使用多种方法生成3D构象,提高成功率"""
        if mol is None:
            return None            
        # 添加氢原子
        mol = Chem.AddHs(mol)        
        # 方法1: ETKDG v3 
        for attempt in range(max_attempts):
            seed = 42 + attempt  # 每次尝试使用不同的随机种子；不同种子--不同构象
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.numThreads = 4  # 利用多线程
            params.useSmallRingTorsions = True
            params.useBasicKnowledge = True
            params.enforceChirality = True            
            if AllChem.EmbedMolecule(mol, params) == 0:  # 0表示成功
                # 力场优化
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)  # MMFF力场
                    return mol #优化之后分子返回
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)  # UFF力场
                        return mol
                    except:
                        continue          
        # 方法2: 基础ETKDG
        #useBasicKnowledge=True 使用基本化学知识
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) == 0: #嵌入成功
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)#就力场优化
                return mol
            except:
                pass                
        # 方法3: 距离几何法
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, useBasicKnowledge=True) == 0:
            return mol            
        # 所有方法都失败,构象生成失败
        return None

    def check_valid_3d_coords(self, pdb_path):
        """验证PDB文件包含有效的3D坐标"""
        atom_count = 0
        nonzero_coords = False
        
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_count += 1
                    # 获取坐标 (x, y, z位于第7-9列)
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        # 检查坐标是否全为0或接近0
                        if abs(x) > 0.01 or abs(y) > 0.01 or abs(z) > 0.01:
                            nonzero_coords = True
                    except:
                        continue
        
        # 有足够多的原子且至少有一组非零坐标
        return atom_count > 3 and nonzero_coords
        
    def parse_vina_output(self, output_file):
        """解析Vina输出文件获取对接分数"""
        try:
            if not os.path.exists(output_file):
                return None
                
            results = []
            with open(output_file, 'r') as f:
                for line in f:
                    if line.startswith('   ') and not line.startswith('      '):
                        # Vina输出格式为: "   1     -8.7      0.000      0.000"
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                mode = parts[0]
                                score = float(parts[1])
                                # 每行存储为 [分数, 模式]
                                results.append([score, mode])
                            except:
                                continue
            
            return results if results else None
        except Exception as e:
            logging.error(f"解析Vina输出失败: {str(e)}")
            return None

    def prepare_receptor(self):
        """准备受体蛋白文件"""
        logging.info("准备受体蛋白...")
        
        # 转换受体文件
        receptor_pdbqt = self.receptor_pdb + "qt"
        if not os.path.exists(receptor_pdbqt):
            self.converter.convert_receptor_pdb_files_to_pdbqt(
                self.receptor_pdb,
                self.vars["mgl_python"],
                self.vars["prepare_receptor4.py"],
                self.vars["number_of_processors"]
            )
            
        return receptor_pdbqt

    def prepare_ligands_batch(self, smi_file):
        """使用autogrow的方法批量准备配体分子"""
        logging.info("批量准备配体分子...")
        total_start_time = time.time()

        # 1. SMILES转3D SDF
        logging.info("Stage 1: SMILES to 3D SDF conversion starting...")
        stage1_start_time = time.time()
        if not os.path.exists(self.sdf_dir):
            os.makedirs(self.sdf_dir)
        convert_to_3d(self.vars, smi_file, self.sdf_dir) # Ensure SDFs go to self.sdf_dir
        logging.info(f"Stage 1: SMILES to 3D SDF conversion finished. Duration: {time.time() - stage1_start_time:.2f} seconds.")

        # 2. SDF转PDB
        logging.info("Stage 2: SDF to PDB conversion starting...")
        stage2_start_time = time.time()
        convert_sdf_to_pdbs(self.vars, self.sdf_dir, self.sdf_dir) 
        actual_pdb_output_dir = os.path.join(self.vars["output_directory"], "PDBs")
        if not os.path.exists(actual_pdb_output_dir):
             logging.warning(f"PDB output directory {actual_pdb_output_dir} not found after SDF to PDB conversion. This might lead to errors.")
        logging.info(f"Stage 2: SDF to PDB conversion finished. Duration: {time.time() - stage2_start_time:.2f} seconds.")

        # 只保留每个SMILES的第一个PDB, using the correct PDB directory
        logging.info("Filtering PDB files (keep one per SMILES)...")
        filter_pdb_start_time = time.time()
        keep_one_pdb_per_smiles(actual_pdb_output_dir)
        logging.info(f"Filtering PDB files finished. Duration: {time.time() - filter_pdb_start_time:.2f} seconds.")

        # 3. PDB转PDBQT（并行处理）
        logging.info("Stage 3: PDB to PDBQT conversion starting...")
        stage3_start_time = time.time()
        pdb_files_to_convert = glob.glob(os.path.join(actual_pdb_output_dir, "*.pdb"))
        pdbqt_files = []

        if not pdb_files_to_convert:
            logging.warning(f"No PDB files found in {actual_pdb_output_dir} for PDBQT conversion.")
        else:
            logging.info(f"Starting PDB to PDBQT conversion for {len(pdb_files_to_convert)} files using {self.number_of_processors} workers.")
            with ProcessPoolExecutor(max_workers=self.number_of_processors) as pool:
                future_to_pdb = {
                    pool.submit(
                        self._convert_single_pdb_to_pdbqt_worker, 
                        pdb_file, 
                        self.vars["mgl_python"], 
                        self.vars["prepare_ligand4.py"]
                    ): pdb_file for pdb_file in pdb_files_to_convert
                }
                for future in tqdm(as_completed(future_to_pdb), total=len(pdb_files_to_convert), desc="PDB to PDBQT"):
                    pdb_file_path = future_to_pdb[future]
                    try:
                        result_pdbqt_file = future.result()
                        if result_pdbqt_file:
                            pdbqt_files.append(result_pdbqt_file)
                    except Exception as exc:
                        logging.error(f'{pdb_file_path} generated an exception during PDBQT conversion: {exc}')
        
        if not pdbqt_files:
            raise RuntimeError("没有成功生成任何PDBQT文件,无法进行后续对接。请检查MGLTools配置和PDB文件内容。")
        
        logging.info(f"Stage 3: PDB to PDBQT conversion finished. Duration: {time.time() - stage3_start_time:.2f} seconds. {len(pdbqt_files)} PDBQT files generated.")
        logging.info(f"Total ligand preparation time: {time.time() - total_start_time:.2f} seconds.")
        return pdbqt_files

    def _convert_single_pdb_to_pdbqt_worker(self, pdb_file_path, mgl_python_path, prep_ligand_script_path):
        """Worker function to convert a single PDB to PDBQT."""
        pdbqt_file = pdb_file_path + "qt"
        # Check if already converted or source PDB exists
        if not os.path.exists(pdb_file_path):
            logging.warning(f"Source PDB file does not exist: {pdb_file_path}")
            return None
            
        if os.path.exists(pdbqt_file): # If already exists, skip conversion but return path
            return pdbqt_file

        cmd = [
            mgl_python_path,
            prep_ligand_script_path,
            "-l", pdb_file_path,
            "-o", pdbqt_file
        ]
        try:
            # It's important to capture output here if there are common, non-fatal errors from MGLTools
            # For now, DEVNULL as per original script for successful runs.
            process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if os.path.exists(pdbqt_file):
                return pdbqt_file
            else:
                # This case (check=True but file not existing) should be rare but log it.
                logging.error(f"PDBQT conversion ran for {pdb_file_path} but output {pdbqt_file} not found. STDOUT: {process.stdout} STDERR: {process.stderr}")
                return None
        except subprocess.CalledProcessError as e:
            logging.error(f"PDBQT conversion failed for: {pdb_file_path}. Error: {e.stderr}")
            return None
        except Exception as e: # Catch other potential errors
            logging.error(f"An unexpected error occurred during PDBQT conversion for {pdb_file_path}: {str(e)}")
            return None

    def run_docking_batch(self, receptor_pdbqt, pdbqt_files):
        """批量执行对接"""
        logging.info(f"开始批量对接 {len(pdbqt_files)} 个分子...")
        docking_start_time = time.time()

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        max_workers = self.number_of_processors # Use configured number of processors
        futures = []
        scores = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ligand_file in pdbqt_files:
                futures.append(executor.submit(
                    vina_dock_single, ligand_file, receptor_pdbqt, self.results_dir, self.vars
                ))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="对接进度"):
                ligand_file, success, score = future.result()
                if success and score is not None:
                    mol_name = os.path.basename(ligand_file).replace(".pdbqt", "")
                    scores[mol_name] = score
        
        logging.info(f"批量对接完成. Duration: {time.time() - docking_start_time:.2f} seconds. Successfully docked {len(scores)} molecules.")
        return scores

    def process_ligand(self, smile):
        pdb_path = None
        try:
            # 记录开始处理
            logging.info(f"开始处理分子: {smile}")
            
            # 生成分子对象
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                logging.warning(f"无法从SMILES生成分子: {smile}")
                return None
            
            # 记录分子信息
            logging.info(f"分子信息 - 原子数: {mol.GetNumAtoms()}, 键数: {mol.GetNumBonds()}")
            
            # 处理特殊情况：分子过大或过于复杂
            if mol.GetNumAtoms() > 100:
                logging.warning(f"分子过大，跳过: {smile} (原子数: {mol.GetNumAtoms()})")
                return None
                
            # 改进的3D构象生成
            mol_3d = self.generate_3d_conformer(mol)
            if mol_3d is None:
                logging.warning(f"无法生成3D构象: {smile}")
                return None
                
            # 记录3D构象生成成功
            logging.info(f"成功生成3D构象: {smile}")
                
            # 转换为PDB格式
            pdb_path = os.path.join(self.output_dir, f"temp_{hash(smile)}.pdb")
            Chem.MolToPDBFile(mol_3d, pdb_path)
            
            # 验证PDB文件包含有效的3D坐标
            if not self.check_valid_3d_coords(pdb_path):
                logging.error(f"生成的PDB缺少有效的3D坐标: {smile}")
                return None
            
            # 记录PDB转换成功
            logging.info(f"成功生成PDB文件: {pdb_path}")
                
            # 转换为PDBQT格式
            try:
                pdbqt_path = pdb_path + "qt"
                cmd = [
                    self.vars["mgl_python"],
                    self.vars["prepare_ligand4.py"],
                    "-l", pdb_path,
                    "-o", pdbqt_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"成功转换为PDBQT格式: {pdbqt_path}")
            except Exception as e:
                logging.error(f"PDBQT转换失败: {str(e)}")
                os.rename(pdb_path, f"{pdb_path}.error")
                return None
                
            if not os.path.exists(pdbqt_path):
                logging.error(f"配体转换失败 - 没有输出文件: {smile}")
                return None
        
            # 执行对接
            logging.info(f"开始执行对接: {smile}")
            out_file = os.path.join(self.results_dir, f"temp_{hash(smile)}_out.pdbqt")
            log_file = os.path.join(self.results_dir, f"temp_{hash(smile)}.log")
            
            cmd = [
                self.vars["docking_executable"],
                "--receptor", self.receptor_pdb + "qt",
                "--ligand", pdbqt_path,
                "--center_x", str(self.vars["center_x"]),
                "--center_y", str(self.vars["center_y"]),
                "--center_z", str(self.vars["center_z"]),
                "--size_x", str(self.vars["size_x"]),
                "--size_y", str(self.vars["size_y"]),
                "--size_z", str(self.vars["size_z"]),
                "--out", out_file,
                "--log", log_file
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 关键修复: 从 out_file (PDBQT) 而不是 log_file 读取分数
            score = extract_vina_score(out_file)
            
            if score is None:
                logging.warning(f"无法从对接输出获取分数: {smile}")
                return None
                
            logging.info(f"对接成功完成: {smile}, 最佳分数: {score}")
            return score
            
        except Exception as e:
            logging.error(f"对接失败，分子: {smile}，错误: {str(e)}")
            return None
        finally:
            # 清理临时文件
            if pdb_path is not None:
                for ext in ['', 'qt', '.error', '.sdf']:
                    path = f"{pdb_path}{ext}"
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
            # 清理对接输出文件
            for file_pattern in [f"temp_{hash(smile)}*"]:
                for file_path in glob.glob(os.path.join(self.results_dir, file_pattern)):
                    try:
                        os.remove(file_path)
                    except:
                        pass

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Molecular Docking Pipeline')
    parser.add_argument('-i', '--input', default="datasets/source_compounds/naphthalene_smiles.smi", help='Input SMILES file')#/data1/tgy/GA_llm/output/generation_0_filtered.smi
    parser.add_argument('-r', '--receptor', default="tutorial/PARP/4r6eA_PARP1_prepared.pdb", help='Receptor PDB file path')#/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb
    parser.add_argument('-o', '--output', default="test_docking_0/initial_generation.smi", help='Output file path')#/data1/tgy/GA_llm/output/docking_results/generation_o_docked.smi
    parser.add_argument('-m', '--mgltools', default="mgltools_x86_64Linux2_1.5.6", help='MGLTools installation path')
    parser.add_argument('--max_failures', type=int, default=5, help='最大连续失败次数，超过此数将暂停并提示')
    parser.add_argument('--use_batch', action='store_true', help='使用批量处理模式')
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1, 
                        help='Number of processors to use for parallel tasks. -1 for all available CPU cores.')
    
    # 为对接盒子添加命令行参数
    parser.add_argument('--center_x', type=float, required=True, help='Docking box center X coordinate')
    parser.add_argument('--center_y', type=float, required=True, help='Docking box center Y coordinate')
    parser.add_argument('--center_z', type=float, required=True, help='Docking box center Z coordinate')
    parser.add_argument('--size_x', type=float, required=True, help='Docking box size X dimension')
    parser.add_argument('--size_y', type=float, required=True, help='Docking box size Y dimension')
    parser.add_argument('--size_z', type=float, required=True, help='Docking box size Z dimension')

    # 添加 --multithread_mode 参数
    parser.add_argument('--multithread_mode', default="serial", 
                        choices=["mpi", "multithreading", "serial"],
                        help='Multithreading mode for docking (if applicable internally by VinaDocking or similar classes)')

    args = parser.parse_args()
    
    # Determine the actual number of processors to use
    actual_num_processors = args.number_of_processors
    try:
        cpu_cores = os.cpu_count()
        if not cpu_cores or cpu_cores < 1: # Fallback if cpu_count is None or invalid
            cpu_cores = 1 
    except NotImplementedError:
        cpu_cores = 1 # Fallback if os.cpu_count() is not implemented
        
    if args.number_of_processors == -1:
        actual_num_processors = cpu_cores
        logging.info(f"Using all available processors: {actual_num_processors}")
    elif args.number_of_processors > cpu_cores:
        logging.warning(f"Requested {args.number_of_processors} processors, but only {cpu_cores} are available. Using {cpu_cores}.")
        actual_num_processors = cpu_cores
    elif args.number_of_processors < 1:
        logging.warning(f"Requested {args.number_of_processors} processors, which is invalid. Using 1 processor.")
        actual_num_processors = 1
    else:
        actual_num_processors = args.number_of_processors
        logging.info(f"Using {actual_num_processors} processors as specified.")

    # 准备输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    setup_logging(os.path.dirname(args.output))
    
    # 初始化对接执行器
    executor = DockingExecutor(
        receptor_pdb=args.receptor,
        output_dir=os.path.dirname(args.output),
        mgltools_path=args.mgltools,
        number_of_processors=actual_num_processors,
        # 传递对接盒子参数
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z
    )
    
    # 读取输入文件
    with open(args.input) as f:
        smiles_list = [line.strip().split()[0] for line in f if line.strip()]
    
    # 准备受体
    receptor_pdbqt = executor.prepare_receptor()
    
    if args.use_batch:
        logging.info("使用批量处理模式...")
        # 将SMILES列表写入临时文件
        temp_smi_file = os.path.join(executor.output_dir, "temp_smiles.smi")
        with open(temp_smi_file, 'w') as f:
            for i, smile in enumerate(smiles_list):
                f.write(f"{smile} naphthalene_{i+1}\n")
        
        try:
            # 批量准备配体
            pdbqt_files = executor.prepare_ligands_batch(temp_smi_file)
            
            # 批量执行对接
            scores_dict = executor.run_docking_batch(receptor_pdbqt, pdbqt_files)
            
            # 整理结果
            results = []
            for i, smile in enumerate(smiles_list):
                mol_name = f"naphthalene_{i+1}"
                score = scores_dict.get(mol_name)
                results.append(score)
        except Exception as e:
            logging.error(f"批量处理失败: {str(e)}")
            logging.info("切换到单分子处理模式...")
            args.use_batch = False
    
    if not args.use_batch:
        # 单分子处理模式
        logging.info("使用单分子处理模式...")
        results = []
        consecutive_failures = 0
        
        for i, smile in enumerate(tqdm(smiles_list, desc="对接进度")):
            result = executor.process_ligand(smile)
            results.append(result)
            
            # 检查是否连续失败
            if result is None:
                consecutive_failures += 1
                if consecutive_failures >= args.max_failures:
                    logging.warning(f"连续失败 {consecutive_failures} 次，请检查对接配置")
                    consecutive_failures = 0  # 重置计数器
            else:
                consecutive_failures = 0
                
            # 每处理50个分子保存一次中间结果
            if (i + 1) % 50 == 0:
                with open(f"{args.output}.partial", 'w') as f:
                    for s, r in zip(smiles_list[:i+1], results):
                        if r is not None:
                            f.write(f"{s}\t{r:.2f}\n")
                logging.info(f"已完成 {i+1}/{len(smiles_list)} 分子对接，中间结果已保存")
    
    # 写入结果文件，按分数排序（分数越低越好排在前面）
    success_count = 0
    total_score = 0.0
    valid_results = []
    
    # 收集有效结果
    for smile, score in zip(smiles_list, results):
        if score is not None:
            success_count += 1
            total_score += score
            valid_results.append((smile, score))
    
    # 按分数排序（分数越低越好，升序排列）
    valid_results.sort(key=lambda x: x[1])
    
    # 写入排序后的结果
    with open(args.output, 'w') as f:
        for smile, score in valid_results:
            f.write(f"{smile}\t{score:.2f}\n")
    
    # 计算平均得分
    average_score = 0.0
    if success_count > 0:
        average_score = total_score / success_count
        
    logging.info(f"对接完成。成功率: {success_count}/{len(smiles_list)} ({success_count/len(smiles_list)*100:.1f}%)。")
    logging.info(f"种群平均对接得分: {average_score:.2f} kcal/mol")
    logging.info(f"结果保存至 {args.output}")

if __name__ == "__main__":
    main()
