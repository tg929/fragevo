#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragEvo 混合工作流执行脚本
==========================
1. 种群初始化和评估
2. 基于父代的分子分解与掩码
3. 使用GPT模型生成新的、多样化的分子
4. 对父代和GPT生成的分子进行遗传算法操作(交叉、突变)
5. 对新生成的子代进行评估
6. 通过选择策略（单目标或多目标）筛选出下一代种群
7. 继续迭代
"""
import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import threading
import queue
import csv
import hashlib
import random
from operations.stating.config_snapshot_generator import save_config_snapshot #保存参数（快照）
import multiprocessing  
import shutil  
from rdkit import Chem
from datasets.decompose.demo_frags import break_into_fragments
from utils.chem_metrics import ChemMetricCache

# 移除全局日志配置，避免多进程日志冲突
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 确保logger有基本的handler，但不会与其他进程冲突
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent #Path(__file__).resolve()：当前脚本目录/地址/data1/ytg/medium_models/FragEvo/operations/operations_execute_fragevo_demo.py  .resolve()：将相对路径转换为绝对路径 
                                                             #整个项目地址：/data1/ytg/medium_models/FragEvo
sys.path.insert(0, str(PROJECT_ROOT))#0：添加目录到搜索列表最前面


class FragEvoWorkflowExecutor:    #工作流；主函数/入口文件就是在调用这个类
    def __init__(self, config_path: str, receptor_name: Optional[str] = None, output_dir_override: Optional[str] = None, num_processors_override: Optional[int] = None):
        """
        初始化FragEvo工作流执行器。        
        Args:
            config_path (str): 配置文件路径。
            receptor_name (Optional[str]): 目标受体名称。如果为None, 则使用默认受体。
            output_dir_override (Optional[str]): 覆盖配置文件中的输出目录。
            num_processors_override (Optional[int]): 覆盖配置文件中的处理器数量。
        """
        self.config_path = config_path
        self.config = self._load_config()        
        # 应用处理器数量覆盖
        if num_processors_override is not None:#有自定义处理器数量设置
            self.config['performance']['number_of_processors'] = num_processors_override
            logger.info(f"运行时覆盖处理器数量为: {num_processors_override}")
            
        self.run_params = {}
        self._setup_parameters_and_paths(receptor_name, output_dir_override)
        self.metric_cache = ChemMetricCache(self.output_dir / "chem_metric_cache.json")
        self._save_run_parameters()
        self.lineage_tracker_path: Optional[Path] = (
            self.output_dir / "lineage_tracker.json" if self.enable_lineage_tracking else None
        )
        self.lineage_tracker = self._load_lineage_tracker()
        self.history_paths: Dict[str, str] = {}
        self.smiles_to_history: Dict[str, str] = {}
        self.history_records: Dict[str, Dict] = {}
        self.removed_history_records: Dict[str, Dict] = {}
        self.current_active_histories: Set[str] = set()
        self.history_root_counter = 0
        self.placeholder_roots: Dict[int, str] = {}
        self.last_offspring_histories: Set[str] = set()
        self.last_offspring_smiles: Set[str] = set()
        logger.info(f"FragEvo工作流初始化完成, 输出目录: {self.output_dir}")
        logger.info(f"最大迭代代数: {self.max_generations}")

    def _load_config(self) -> dict:#加载配置文件
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)       

    def _setup_parameters_and_paths(self, receptor_name: Optional[str], output_dir_override: Optional[str]):        
        self.project_root = Path(self.config.get('paths', {}).get('project_root', PROJECT_ROOT))
        workflow_config = self.config.get('workflow', {})
        seed_value = workflow_config.get("seed", 42)
        try:
            self.seed = int(seed_value)
        except (TypeError, ValueError):
            self.seed = 42
        random.seed(self.seed)
        self.run_params["seed"] = self.seed
        gpt_config = self.config.get('gpt', {})
        self.dynamic_masking_config = gpt_config.get('dynamic_masking', {'enable': False})
        self.enable_lineage_tracking = bool(workflow_config.get("enable_lineage_tracking", False))
        self.run_params["enable_lineage_tracking"] = self.enable_lineage_tracking
        # 记录配置和根目录
        self.run_params['config_file_path'] = self.config_path
        self.run_params['project_root'] = str(self.project_root)
        # 确定输出目录
        if output_dir_override:
            output_dir_name = output_dir_override
        else:
            output_dir_name = workflow_config.get('output_directory', 'FragEvo_output')
        base_output_dir = self.project_root / output_dir_name
        self.run_params['base_output_dir'] = str(base_output_dir)
        # 根据受体确定最终运行目录
        self.receptor_name = receptor_name
        if self.receptor_name:
            self.output_dir = base_output_dir / self.receptor_name
            self.run_params['receptor_name'] = self.receptor_name
        else:
            # 如果没有指定受体，使用默认或创建一个通用运行目录
            default_receptor_info = self.config.get('receptors', {}).get('default_receptor', {})
            default_receptor_name = default_receptor_info.get('name', 'default_run')
            self.output_dir = base_output_dir / default_receptor_name
            self.run_params['receptor_name'] = default_receptor_name
        self.run_params['run_specific_output_dir'] = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 加载GA和GPT的核心参数
        self.max_generations = workflow_config.get('max_generations', 10)
        self.initial_population_file = workflow_config.get('initial_population_file')
        self.run_params['max_generations'] = self.max_generations
        self.run_params['initial_population_file'] = self.initial_population_file
        # 记录选择模式
        selection_config = self.config.get('selection', {})
        self.run_params['selection_mode'] = selection_config.get('selection_mode', 'single_objective')
    def _get_dynamic_mask_count(self, generation: int) -> int:
        """
        根据当前代数计算动态掩码片段的数量。
        如果未启用动态掩码，则返回配置中的固定值。
        Args:
            generation (int): 当前的进化代数。            
        Returns:
            int: 应该用于掩码的片段数量。
        """
        if not self.dynamic_masking_config.get('enable', False) or self.max_generations <= 1:
            # 如果不启用或总代数只有1代，则使用固定的值
            return self.config.get('gpt', {}).get('n_fragments_to_mask', 1)        
        initial_mask = self.dynamic_masking_config.get('initial_mask_fragments', 2)
        final_mask = self.dynamic_masking_config.get('final_mask_fragments', 1)        
        # 使用线性插值计算当前代数的掩码数
        # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        # 这里 x=generation, x1=1, y1=initial_mask, x2=max_generations, y2=final_mask        
        # 防止除以零
        if self.max_generations == 1:
            return initial_mask            
        progress = (generation - 1) / (self.max_generations - 1)
        mask_count = initial_mask + progress * (final_mask - initial_mask)        
        # 四舍五入到最近的整数，并确保结果在[final_mask, initial_mask]范围内
        return int(round(max(min(mask_count, initial_mask), final_mask)))
    def _save_run_parameters(self):
        """保存本次运行的完整参数快照。"""
        snapshot_file_path = self.output_dir / "execution_config_snapshot.json"
        success = save_config_snapshot(
            original_config=self.config,
            execution_context=self.run_params,
            output_file_path=str(snapshot_file_path)
        )
        if success:
            logger.info(f"完整的执行配置快照已保存到: {snapshot_file_path}")
        else:
            logger.error("保存执行配置快照失败")
    def _load_lineage_tracker(self) -> Dict[str, List[Dict]]:
        """从磁盘加载既有的血统记录。"""
        if not getattr(self, "enable_lineage_tracking", False):
            return {}
        if self.output_dir and hasattr(self, "output_dir"):
            path = getattr(self, "lineage_tracker_path", None)
        else:
            path = None
        if path and path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
                logger.warning("血统跟踪文件格式异常，已忽略原有记录。")
            except Exception as exc:
                logger.warning(f"无法加载血统跟踪文件 {path}: {exc}")
        return {}
    def _save_lineage_tracker(self) -> None:
        """将血统跟踪记录持久化到磁盘。"""
        if not getattr(self, "enable_lineage_tracking", False):
            return
        if not self.lineage_tracker_path:
            return
        try:
            with open(self.lineage_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(self.lineage_tracker, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"保存血统跟踪文件失败: {exc}")
    def _write_jsonl(self, output_path: Path, entries: List[Dict]) -> None:
        """将记录写入 JSONL 文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in entries:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    def _read_jsonl(self, input_path: Path) -> List[Dict]:
        """读取 JSONL 文件并返回字典列表。"""
        if not input_path or not input_path.exists():
            return []
        entries: List[Dict] = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析血统记录: {line}")
        except Exception as exc:
            logger.warning(f"读取血统文件 {input_path} 失败: {exc}")
        return entries
    def _read_smiles_from_file(self, file_path: Path, first_column_only: bool = True) -> List[str]:
        """读取SMILES文件，默认只返回第一列。"""
        smiles: List[str] = []
        if not file_path or not file_path.exists():
            return smiles
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    smiles.append(parts[0] if first_column_only else parts)
        except Exception as exc:
            logger.warning(f"读取文件 {file_path} 时发生错误: {exc}")
        return smiles

    def _tokenize_smiles_sequence(self, smiles: str, include_eos: bool = True) -> Optional[str]:
        """将单个SMILES转换为[BOS]/[SEP]/[EOS]格式的片段序列。"""
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        frag_count: Optional[int] = None
        fragments: List[str] = []
        if mol:
            try:
                result = break_into_fragments(mol, smiles)
                fragments_candidate = None
                if isinstance(result, tuple):
                    if len(result) >= 3:
                        fragments_candidate = result[1]
                        frag_count = result[2]
                    if (not fragments_candidate or isinstance(fragments_candidate, float)) and len(result) >= 4:
                        fragments_candidate = result[3]
                if isinstance(fragments_candidate, list):
                    fragments = [frag for frag in fragments_candidate if isinstance(frag, str) and frag]
                elif isinstance(fragments_candidate, str):
                    fragments = [frag for frag in fragments_candidate.split() if frag]
                else:
                    fragments = []
                if frag_count is None or frag_count <= 1:
                    fragments = [smiles] if not fragments else fragments
            except Exception:
                fragments = [smiles]
        else:
            fragments = [smiles]
        if not fragments:
            fragments = [smiles]
        joined = "[SEP]".join(fragments)
        if include_eos:
            return f"[BOS]{joined}[EOS]"
        return f"[BOS]{joined}[SEP]"

    def _generate_tokenized_file(
        self,
        source_path: Optional[str],
        output_path: Path,
        include_eos: bool = True,
        first_column_only: bool = True
    ) -> None:
        """根据输入SMILES生成带有序列标记的SMI文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_path:
            output_path.touch()
            return
        src_path = Path(source_path)
        if not src_path.exists():
            output_path.touch()
            return
        lines: List[str] = []
        try:
            with open(src_path, 'r', encoding='utf-8') as src:
                for line in src:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if first_column_only:
                        parts = stripped.split()
                        if not parts:
                            continue
                        stripped = parts[0]
                    tokenized = self._tokenize_smiles_sequence(stripped, include_eos=include_eos)
                    if tokenized:
                        lines.append(tokenized)
        except Exception as exc:
            logger.warning(f"读取来源文件 {src_path} 以生成序列格式时失败: {exc}")
            output_path.touch()
            return
        try:
            with open(output_path, 'w', encoding='utf-8') as dst:
                for entry in lines:
                    dst.write(entry + '\n')
        except Exception as exc:
            logger.warning(f"写入序列文件 {output_path} 时失败: {exc}")

    def _copy_pre_tokenized_file(self, source_path: Optional[str], output_path: Path) -> None:
        """将已经是标记序列的文件复制到目标路径。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_path:
            output_path.touch()
            return
        src_path = Path(source_path)
        if not src_path.exists():
            output_path.touch()
            return
        try:
            with open(src_path, 'r', encoding='utf-8') as src, open(output_path, 'w', encoding='utf-8') as dst:
                dst.writelines(src.readlines())
        except Exception as exc:
            logger.warning(f"复制标记序列文件 {src_path} 时失败: {exc}")
            output_path.touch()

    def _export_tokenized_representations(
        self,
        generation: int,
        parent_smiles_file: Path,
        masked_fragments_file: Optional[str],
        gpt_generated_file: Optional[str]
    ) -> None:
        """在@gpt_generated目录下输出父代、GPT子代和掩码片段的序列表示。"""
        gen_dir = self.output_dir / f"generation_{generation}"
        gpt_dir = gen_dir / "gpt_generated"
        gpt_dir.mkdir(parents=True, exist_ok=True)
        self._generate_tokenized_file(
            str(parent_smiles_file),
            gpt_dir / "initial_population.smi",
            include_eos=True,
            first_column_only=True
        )
        self._generate_tokenized_file(
            gpt_generated_file,
            gpt_dir / "gpt_generated_molecules_tokenized.smi",
            include_eos=True,
            first_column_only=True
        )
        self._copy_pre_tokenized_file(
            masked_fragments_file,
            gpt_dir / "masked_fragments.smi"
        )

    def _update_lineage_tracker(self, lineage_entries: List[Dict]) -> None:
        """更新内存中的血统跟踪数据并同步到磁盘。"""
        if not self.enable_lineage_tracking:
            return
        if not lineage_entries:
            return
        for entry in lineage_entries:
            child = entry.get("child")
            if not child:
                continue
            history = self.lineage_tracker.setdefault(child, [])
            history.append({
                "generation": entry.get("generation"),
                "sources": entry.get("sources", [])
            })
        self._save_lineage_tracker()

    def _record_initial_population(self, formatted_file: Path) -> None:
        """记录初代种群的血统来源。"""
        if not self.enable_lineage_tracking:
            return
        smiles_list = self._read_smiles_from_file(formatted_file)
        if not smiles_list:
            return
        entries: List[Dict] = []
        for smi in smiles_list:
            self._ensure_history(smi, generation=0)
            history = self.lineage_tracker.get(smi)
            if history:
                continue
            sources = [{
                "operation": "initial_population",
                "parents": []
            }]
            self.lineage_tracker.setdefault(smi, []).append({
                "generation": 0,
                "sources": sources
            })
            entries.append({
                "generation": 0,
                "child": smi,
                "sources": sources
            })
        if entries:
            lineage_file = formatted_file.parent / "initial_population_lineage.jsonl"
            self._write_jsonl(lineage_file, entries)
            self._save_lineage_tracker()
            logger.info(f"初代种群血统记录已保存到: {lineage_file}")
    def _short_hash(self, value: str) -> str:
        return hashlib.md5(value.encode('utf-8')).hexdigest()[:6]
    def _register_history(self, smiles: str, history: str) -> None:
        self.history_paths[smiles] = history
        self.smiles_to_history[smiles] = history
    def _create_root_history(self, smiles: str) -> str:
        if smiles in self.smiles_to_history:
            return self.smiles_to_history[smiles]
        token = f"ROOT-{self.history_root_counter}"
        self.history_root_counter += 1
        history = token
        self._register_history(smiles, history)
        return history
    def _ensure_history(self, smiles: str, generation: Optional[int] = None) -> str:
        history = self.smiles_to_history.get(smiles)
        if history:
            return history
        return self._create_root_history(smiles)
    def _create_generation_placeholder_root(self, generation: int) -> str:
        placeholder = self.placeholder_roots.get(generation)
        if placeholder is None:
            placeholder = f"GEN{generation}-ROOT"
            self.placeholder_roots[generation] = placeholder
        return placeholder
    def _build_operation_token(self, operation: str, parents: List[str], generation: int) -> str:
        op = (operation or "GEN").upper()
        parent_ids = []
        for parent in parents:
            parent_history = self.smiles_to_history.get(parent)
            if not parent_history:
                parent_history = self._ensure_history(parent, generation=generation)
            parent_ids.append(self._short_hash(parent_history))
        if not parent_ids:
            parent_ids.append(f"G{generation}")
        return f"{op}-{'_'.join(parent_ids)}"
    def _derive_history(self, smiles: str, generation: int, sources: List[Dict]) -> str:
        existing = self.smiles_to_history.get(smiles)
        if existing:
            return existing
        parent_history = None
        op_tokens: List[str] = []
        if sources:
            for source in sources:
                operation = source.get("operation", "GEN")
                parents = source.get("parents") or []
                if parent_history is None and parents:
                    for parent in parents:
                        parent_history = self.smiles_to_history.get(parent)
                        if parent_history:
                            break
                op_tokens.append(self._build_operation_token(operation, parents, generation))
            if parent_history is None and sources[0].get("parents"):
                first_parent = sources[0]["parents"][0]
                parent_history = self._ensure_history(first_parent, generation=generation)
        if parent_history is None:
            parent_history = self._create_generation_placeholder_root(generation)
        if not op_tokens:
            op_tokens.append(self._build_operation_token("GEN", [], generation))
        history = f"{parent_history}|{'_'.join(op_tokens)}"
        self._register_history(smiles, history)
        return history
    def _assign_histories_to_offspring(self, generation: int, lineage_entries: List[Dict]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entry in lineage_entries:
            child = entry.get("child")
            if not child:
                continue
            history = self._derive_history(child, generation, entry.get("sources", []))
            entry["history_data"] = history
            mapping[child] = history
        return mapping
    def _compute_metrics(self, smiles: str, docking_score: Optional[float]) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = {
            "docking_score": docking_score,
            "total": docking_score
        }
        qed, sa = self.metric_cache.get_or_compute(smiles)
        metrics["qed"] = qed
        metrics["sa"] = sa
        return metrics
    def _upsert_history_record(self, history: str, smiles: str, generation: int, docking_score: Optional[float], mark_active: bool) -> None:
        record = self.history_records.get(history, {
            "smiles": smiles,
            "history_data": history,
            "generation_created": generation,
            "status": "inactive"
        })
        metrics = self._compute_metrics(smiles, docking_score)
        record["smiles"] = smiles
        record["history_data"] = history
        record.setdefault("generation_created", generation)
        record["last_generation"] = generation
        record["metrics"] = metrics
        record["docking_score"] = docking_score
        if mark_active:
            record["status"] = "active"
        elif record.get("status") not in ("removed", "active"):
            record["status"] = "inactive"
        self.history_records[history] = record
    def _ingest_population_metrics(self, docked_file: Path, generation: int, mark_active: bool = False) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        path_obj = Path(docked_file)
        if not path_obj.exists():
            return mapping
        with open(path_obj, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                smiles = parts[0]
                docking_score: Optional[float] = None
                if len(parts) >= 2:
                    try:
                        docking_score = float(parts[1])
                    except ValueError:
                        docking_score = None
                history = self._ensure_history(smiles, generation=generation)
                mapping[smiles] = history
                self._upsert_history_record(history, smiles, generation, docking_score, mark_active)
        self.metric_cache.flush()
        return mapping
    def _mark_histories_active(self, histories: Set[str], generation: int) -> None:
        for history in histories:
            record = self.history_records.get(history)
            if not record:
                continue
            record["status"] = "active"
            record["last_generation"] = generation
            self.history_records[history] = record
        self.current_active_histories = set(histories)
    def _mark_histories_removed(self, histories: Set[str], generation: int) -> None:
        for history in histories:
            record = self.history_records.get(history)
            if not record:
                continue
            if record.get("status") == "removed":
                continue
            record["status"] = "removed"
            record["removed_generation"] = generation
            self.history_records[history] = record
            self.removed_history_records[history] = record
    def _format_float(self, value: Optional[float]) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.6f}"
        except Exception:
            return ""
    def _export_evomo_files(self) -> None:
        pop_file = self.output_dir / "pop.csv"
        removed_file = self.output_dir / "removed_ind_act_history.csv"
        pop_headers = ["smiles", "total", "qed", "sa", "docking_score", "history_data"]
        with open(pop_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(pop_headers)
            for history, record in sorted(self.history_records.items()):
                if record.get("status") != "active":
                    continue
                metrics = record.get("metrics", {})
                writer.writerow([
                    record.get("smiles", ""),
                    self._format_float(metrics.get("total")),
                    self._format_float(metrics.get("qed")),
                    self._format_float(metrics.get("sa")),
                    self._format_float(metrics.get("docking_score")),
                    record.get("history_data", history)
                ])
        with open(removed_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["history_data", "total", "qed", "sa", "docking_score", "smiles"])
            for history, record in sorted(self.removed_history_records.items()):
                metrics = record.get("metrics", {})
                writer.writerow([
                    record.get("history_data", history),
                    self._format_float(metrics.get("total")),
                    self._format_float(metrics.get("qed")),
                    self._format_float(metrics.get("sa")),
                    self._format_float(metrics.get("docking_score")),
                    record.get("smiles", "")
                ])

    def _run_script(self, script_path: str, args: List[str]) -> bool:
        """
        统一的脚本执行函数，通过流式处理输出防止死锁，并增加超时保护。
        
        Args:
            script_path (str): 相对于项目根目录的脚本路径。
            args (List[str]): 脚本的命令行参数列表。
            
        Returns:
            bool: 脚本是否执行成功。
        """
        full_script_path = self.project_root / script_path
        cmd = ['python', str(full_script_path)] + args
        logger.debug(f"Executing command: {' '.join(cmd)}")

        env = os.environ.copy()
        seed_value = str(getattr(self, "seed", 42))
        env["PYTHONHASHSEED"] = seed_value
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=str(self.project_root),
                env=env,
                close_fds=True
            ) as process:
                
                # 创建队列来从线程中接收输出
                q_stdout = queue.Queue()
                q_stderr = queue.Queue()

                # 创建并启动线程来实时读取输出
                thread_stdout = threading.Thread(target=self._read_stream, args=(process.stdout, q_stdout))
                thread_stderr = threading.Thread(target=self._read_stream, args=(process.stderr, q_stderr))
                thread_stdout.start()
                thread_stderr.start()

                # 等待进程结束，设置超时
                try:
                    process.wait(timeout=3600)  # 1小时超时
                except subprocess.TimeoutExpired:
                    logger.error(f"Script {script_path} timed out (1 hour). Terminating...")
                    process.kill()  # 强制杀死进程
                    # 再等待一小段时间确保线程能读取完最后的信息
                    thread_stdout.join(timeout=5)
                    thread_stderr.join(timeout=5)
                    # 记录日志并返回失败
                    self._log_subprocess_output(script_path, q_stdout, q_stderr, "after timeout")
                    return False
                
                # 进程正常结束后，等待读取线程完成
                thread_stdout.join()
                thread_stderr.join()

                # 收集并记录输出
                stdout_str, stderr_str = self._log_subprocess_output(script_path, q_stdout, q_stderr, "final")

                if process.returncode == 0:
                    logger.info(f"Script {script_path} executed successfully.")
                    return True
                else:
                    logger.error(f"Script {script_path} failed with return code {process.returncode}.")
                    # 在失败时，即使没有stderr，也记录stdout，可能包含线索
                    if stderr_str:
                        logger.error(f"Error output (stderr):\n{stderr_str}")
                    if stdout_str:
                        logger.error(f"Standard output (stdout):\n{stdout_str}")
                    return False
                    
        except Exception as e:
            logger.error(f"An exception occurred while trying to run script {script_path}: {e}", exc_info=True)
            return False

    def _read_stream(self, stream, q: queue.Queue):
        """实时读取流（stdout/stderr）并放入队列"""
        try:
            for line in iter(stream.readline, ''):
                q.put(line)
        finally:
            stream.close()

    def _log_subprocess_output(self, script_path: str, q_stdout: queue.Queue, q_stderr: queue.Queue, context: str) -> Tuple[str, str]:
        """从队列中收集并记录子进程的输出"""
        stdout_lines = []
        while not q_stdout.empty():
            stdout_lines.append(q_stdout.get_nowait())
        stdout_str = "".join(stdout_lines)

        stderr_lines = []
        while not q_stderr.empty():
            stderr_lines.append(q_stderr.get_nowait())
        stderr_str = "".join(stderr_lines)

        if stdout_str:
            logger.debug(f"--- stdout for {script_path} ({context}) ---\n{stdout_str}")
            logger.debug(f"--- end stdout for {script_path} ---")
        
        return stdout_str, stderr_str

    def _count_molecules(self, file_path: str) -> int:
        """统计SMILES文件中的分子数量"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            return count
        except FileNotFoundError:
            return 0
    def _remove_duplicates_from_smiles_file(self, input_file: str, output_file: str) -> int:
        """
        去除SMILES文件中的重复分子,并为每个分子添加唯一ID。
        输出格式: SMILES  ligand_id_X
        增加文件锁防护，避免并发访问冲突。
        """
        import time
        import random
        
        # 添加随机延迟，避免多进程同时访问文件
        time.sleep(random.uniform(0.1, 0.5))
        
        try:
            unique_smiles = set()
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        smiles = parts[0]
                        if smiles:
                            unique_smiles.add(smiles)            
            unique_smiles_list = sorted(list(unique_smiles))            
            
            # 使用临时文件写入，然后原子性重命名，避免写入冲突
            temp_output_file = output_file + f".tmp_{os.getpid()}_{int(time.time())}"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                for i, smiles in enumerate(unique_smiles_list):
                    f.write(f"{smiles}\tligand_id_{i}\n")
            
            # 原子性重命名
            import shutil
            shutil.move(temp_output_file, output_file)
            
            logger.info(f"去重完成: {len(unique_smiles_list)} 个独特分子保存到 {output_file}")
            return len(unique_smiles_list)
        except Exception as e:
            logger.error(f"去重过程中发生错误: {e}")
            # 清理可能的临时文件
            temp_file = output_file + f".tmp_{os.getpid()}_{int(time.time())}"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return 0

    def _extract_smiles_from_docked_file(self, docked_file: str, output_smiles_file: str) -> bool:
        """从带对接分数的文件中提取纯SMILES,用于遗传操作或分解"""
        try:
            with open(docked_file, 'r') as infile, open(output_smiles_file, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if line:
                        smiles = line.split()[0]
                        outfile.write(f"{smiles}\n")
            return True
        except Exception as e:
            logger.error(f"从 {docked_file} 提取SMILES时出错: {e}")
            return False

    def _execute_ga_stage(
        self,
        ga_op_name: str,
        ga_script: str,
        input_pool_file: str,
        raw_output_file: Path,
        filtered_output_file: Path,
        raw_lineage_file: Path,
        filtered_lineage_file: Path
    ) -> Tuple[bool, Optional[Path]]:
        """辅助函数，用于运行一个GA阶段（如交叉）及其后续的过滤，并返回(是否成功, 过滤后的血统文件路径)。"""
        logger.info(f"开始执行 {ga_op_name}...")
        
        # 运行GA操作
        ga_args = [
            '--smiles_file', input_pool_file,
            '--output_file', str(raw_output_file),
            '--config_file', self.config_path,
            '--seed', str(getattr(self, "seed", 42)),
        ]
        if self.enable_lineage_tracking:
            ga_args.extend(['--lineage_file', str(raw_lineage_file)])

        ga_succeeded = self._run_script(ga_script, ga_args)
        if not ga_succeeded:
            logger.error(f"'{ga_op_name}' 脚本执行失败。")
            return False, None

        # 运行过滤器
        filter_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(raw_output_file),
            '--output_file', str(filtered_output_file)
        ])
        if not filter_succeeded:
            logger.error(f"'{ga_op_name}' 过滤失败。")
            return False, None

        if not self.enable_lineage_tracking:
            logger.info(f"'{ga_op_name}' 操作完成, 生成 {self._count_molecules(str(filtered_output_file))} 个过滤后的分子。")
            return True, None

        filtered_entries = self._filter_lineage_entries(raw_lineage_file, filtered_output_file)
        self._write_jsonl(filtered_lineage_file, filtered_entries)
        logger.info(f"'{ga_op_name}' 操作完成, 生成 {self._count_molecules(str(filtered_output_file))} 个过滤后的分子。")
        return True, filtered_lineage_file

    def _filter_lineage_entries(self, raw_lineage_file: Path, filtered_output_file: Path) -> List[Dict]:
        """根据过滤后的SMILES保留有效的血统记录。"""
        raw_entries = self._read_jsonl(raw_lineage_file)
        if not raw_entries:
            return []
        filtered_smiles = set(self._read_smiles_from_file(filtered_output_file))
        if not filtered_smiles:
            return []
        kept_entries: List[Dict] = []
        for entry in raw_entries:
            child = entry.get("child")
            if not child or child not in filtered_smiles:
                continue
            filtered_entry = dict(entry)
            filtered_entry["child"] = child
            filtered_entry["parents"] = list(filtered_entry.get("parents", []))
            kept_entries.append(filtered_entry)
        return kept_entries

    def run_decomposition_and_masking(self, parent_smiles_file: str, generation: int) -> Optional[str]:
        """
        执行分子分解和掩码操作。        
        Args:
            parent_smiles_file (str): 父代SMILES文件路径。
            generation (int): 当前代数。            
        Returns:
            Optional[str]: 成功则返回掩码后片段文件的路径，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始分解和掩码...")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        masked_fragments_file = gen_dir / "masked_fragments.smi"        
        
        # 检查是否启用动态掩码
        if self.dynamic_masking_config.get('enable', False):
            logger.info(f"第 {generation} 代: 使用动态掩码策略")
            # 动态掩码模式下，参数由被调用的脚本内部处理
            decompose_args = [
                '--input', parent_smiles_file,
                '--output3', str(masked_fragments_file),
                '--current_generation', str(generation),
                '--max_generations', str(self.max_generations),
                '--enable_dynamic_masking' # 添加一个明确的标志
            ]
        else:
            # 固定掩码模式
            n_mask = self._get_dynamic_mask_count(generation)
            logger.info(f"第 {generation} 代: 使用固定掩码数 n_mask = {n_mask}")
            decompose_args = [
                '--input', parent_smiles_file,
                '--output3', str(masked_fragments_file),
                '--mask_fragments', str(n_mask)
            ]
        
        if not self._run_script('datasets/decompose/demo_frags.py', decompose_args):
            logger.error(f"第 {generation} 代: 分解和掩码失败。")
            return None        
        
        if self._count_molecules(str(masked_fragments_file)) == 0:
            logger.warning(f"第 {generation} 代: 未生成任何有效的掩码片段。")
            return None
            
        logger.info(f"第 {generation} 代: 分解和掩码完成，结果保存至 {masked_fragments_file}")
        return str(masked_fragments_file)
    def run_gpt_generation(self, masked_fragments_file: str, generation: int) -> Optional[str]:
        """
        使用GPT模型生成新分子。        
        Args:
            masked_fragments_file (str): 掩码片段文件路径。
            generation (int): 当前代数。            
        Returns:
            Optional[str]: 成功则返回GPT生成的新分子文件路径，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始GPT生成...")
        gen_dir = self.output_dir / f"generation_{generation}"
        gpt_output_dir = gen_dir / "gpt_generated"
        gpt_output_dir.mkdir(exist_ok=True)        
        
        seed = getattr(self, "seed", 42)
        
        # 定义GPT输出文件路径，不再硬编码和移动文件
        gpt_generated_file = gpt_output_dir / "gpt_generated_molecules.smi"
        gpt_args = [
            '--input_file', masked_fragments_file,
            '--seed', str(seed),
            '--output_file', str(gpt_generated_file)  # 直接传递输出路径
        ]

        if not self._run_script('fragmlm/generate_all.py', gpt_args):
            logger.error(f"第 {generation} 代: GPT生成脚本执行失败。")
            return None        

        # 检查指定的输出文件是否已生成且不为空
        generated_count = self._count_molecules(str(gpt_generated_file))
        if generated_count == 0:
            logger.warning(f"第 {generation} 代: GPT生成了0个有效分子。")
            # 不认为是致命错误，可以继续执行GA
            return None
        gpt_smiles = self._read_smiles_from_file(gpt_generated_file)
        placeholder_root = self._create_generation_placeholder_root(generation)
        for smi in gpt_smiles:
            if smi in self.smiles_to_history:
                continue
            op_token = f"GPT-{generation}-{self._short_hash(smi)}"
            history = f"{placeholder_root}|{op_token}"
            self._register_history(smi, history)
            self._upsert_history_record(history, smi, generation, None, mark_active=False)
        logger.info(f"第 {generation} 代: GPT生成完成,产出 {generated_count} 个新分子。")
        return str(gpt_generated_file)
    def _combine_files(self, file_list: List[str], output_file: str) -> bool:
        """合并多个SMILES文件到一个文件"""
        try:
            with open(output_file, 'w') as outf:
                for file_path in file_list:
                    if not file_path or not Path(file_path).exists():
                        continue
                    with open(file_path, 'r') as inf:
                        for line in inf:
                            line = line.strip()
                            if line:
                                outf.write(line + '\n')
            return True
        except Exception as e:
            logger.error(f"合并文件时发生错误: {e}")
            return False
    def run_ga_operations(self, parent_smiles_file: str, gpt_generated_file: Optional[str], generation: int) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
        """
        串行执行遗传算法操作（交叉和突变）以避免死锁。
        
        Args:
            parent_smiles_file (str): 父代SMILES文件路径。
            gpt_generated_file (Optional[str]): GPT生成的SMILES文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[Tuple[str, str, Optional[str], Optional[str]]]: 
                成功则返回 (交叉后代文件, 突变后代文件, 交叉血统文件, 突变血统文件)。
        """
        logger.info(f"第 {generation} 代: 开始串行执行遗传算法操作...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. 合并父代和GPT产出，作为GA操作的输入
        ga_input_pool_file = gen_dir / "ga_input_pool.smi"
        files_to_combine = [parent_smiles_file]
        if gpt_generated_file:
            files_to_combine.append(gpt_generated_file)
        
        if not self._combine_files(files_to_combine, str(ga_input_pool_file)):
            logger.error(f"第 {generation} 代: 合并父代和GPT产出失败。")
            return None
        logger.info(f"第 {generation} 代: GA操作输入池已创建,共 {self._count_molecules(str(ga_input_pool_file))} 个分子。")

        # 2. 串行执行交叉和突变以避免死锁
        crossover_raw_file = gen_dir / "crossover_raw.smi"
        crossover_filtered_file = gen_dir / "crossover_filtered.smi"
        crossover_raw_lineage = gen_dir / "crossover_raw_lineage.jsonl"
        crossover_filtered_lineage = gen_dir / "crossover_filtered_lineage.jsonl"
        mutation_raw_file = gen_dir / "mutation_raw.smi"
        mutation_filtered_file = gen_dir / "mutation_filtered.smi"
        mutation_raw_lineage = gen_dir / "mutation_raw_lineage.jsonl"
        mutation_filtered_lineage = gen_dir / "mutation_filtered_lineage.jsonl"

        # 执行交叉操作
        logger.info(f"第 {generation} 代: 开始交叉操作...")
        crossover_ok, crossover_lineage_path = self._execute_ga_stage(
            "交叉", 'operations/crossover/crossover_demo_finetune.py',
            str(ga_input_pool_file), crossover_raw_file, crossover_filtered_file,
            crossover_raw_lineage, crossover_filtered_lineage
        )
        
        if not crossover_ok:
            logger.error(f"第 {generation} 代: 交叉操作失败。")
            return None

        # 执行变异操作
        logger.info(f"第 {generation} 代: 开始变异操作...")
        mutation_ok, mutation_lineage_path = self._execute_ga_stage(
            "突变", 'operations/mutation/mutation_demo_finetune.py',
            str(ga_input_pool_file), mutation_raw_file, mutation_filtered_file,
            mutation_raw_lineage, mutation_filtered_lineage
        )
        
        if not mutation_ok:
            logger.error(f"第 {generation} 代: 变异操作失败。")
            return None

        logger.info(f"第 {generation} 代: 交叉和变异操作串行完成。")
        return (
            str(crossover_filtered_file),
            str(mutation_filtered_file),
            str(crossover_lineage_path) if crossover_lineage_path else None,
            str(mutation_lineage_path) if mutation_lineage_path else None
        )

    def run_offspring_evaluation(
        self,
        crossover_file: str,
        mutation_file: str,
        generation: int,
        crossover_lineage_file: Optional[str] = None,
        mutation_lineage_file: Optional[str] = None
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        执行子代种群的评估（对接），并生成血统记录。
        
        Args:
            crossover_file (str): 交叉后代文件路径。
            mutation_file (str): 突变后代文件路径。
            generation (int): 当前代数。
            crossover_lineage_file (Optional[str]): 交叉产出对应的血统文件。
            mutation_lineage_file (Optional[str]): 突变产出对应的血统文件。
            
        Returns:
            Optional[Tuple[str, Optional[str]]]: (子代对接结果文件路径, 血统记录文件路径)。
        """
        logger.info(f"第 {generation} 代: 开始子代评估...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. 合并交叉和突变结果
        offspring_raw_file = gen_dir / "offspring_combined_raw.smi"
        if not self._combine_files([crossover_file, mutation_file], str(offspring_raw_file)):
            logger.error(f"第 {generation} 代: 子代合并失败。")
            return None
        
        # 2. 对子代进行去重和格式化（为对接做准备）
        offspring_formatted_file = gen_dir / "offspring_formatted_for_docking.smi"
        offspring_count = self._remove_duplicates_from_smiles_file(
            str(offspring_raw_file), 
            str(offspring_formatted_file)
        )
        offspring_lineage_file = gen_dir / "offspring_lineage.jsonl" if self.enable_lineage_tracking else None
        if offspring_count == 0:
            logger.warning(f"第 {generation} 代: 经过滤和去重后，无有效子代分子。")
            # 创建一个空的对接文件和血统文件，避免后续步骤报错
            offspring_docked_file = gen_dir / "offspring_docked.smi"
            open(offspring_docked_file, 'a').close()
            self.last_offspring_histories = set()
            self.last_offspring_smiles = set()
            if offspring_lineage_file:
                self._write_jsonl(offspring_lineage_file, [])
                return str(offspring_docked_file), str(offspring_lineage_file)
            return str(offspring_docked_file), None

        if self.enable_lineage_tracking and offspring_lineage_file:
            unique_smiles = self._read_smiles_from_file(offspring_formatted_file)
            crossover_entries = self._read_jsonl(Path(crossover_lineage_file)) if crossover_lineage_file else []
            mutation_entries = self._read_jsonl(Path(mutation_lineage_file)) if mutation_lineage_file else []
            offspring_lineage_entries = self._combine_lineage_records(
                generation,
                unique_smiles,
                crossover_entries,
                mutation_entries
            )
            self._assign_histories_to_offspring(generation, offspring_lineage_entries)
            self._write_jsonl(offspring_lineage_file, offspring_lineage_entries)
            self._update_lineage_tracker(offspring_lineage_entries)

        logger.info(f"子代格式化完成: 共 {offspring_count} 个独特分子准备对接。")

        # 3. 对子代进行对接
        offspring_docked_file = gen_dir / "offspring_docked.smi"
        
        # 显式传递处理器数量
        num_processors = self.config.get('performance', {}).get('number_of_processors')
        
        docking_args = [
            '--smiles_file', str(offspring_formatted_file),
            '--output_file', str(offspring_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path,
            '--seed', str(getattr(self, "seed", 42)),
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
        
        # 将处理器数量添加到命令行参数中
        if num_processors is not None:
            docking_args.extend(['--number_of_processors', str(num_processors)])
            
        if not self._run_script('operations/docking/docking_demo_finetune.py', docking_args):
            logger.error(f"第 {generation} 代: 子代对接失败。")
            return None

        docked_count = self._count_molecules(str(offspring_docked_file))
        logger.info(f"第 {generation} 代: 子代评估完成，{docked_count} 个分子已对接。")

        offspring_mapping = self._ingest_population_metrics(offspring_docked_file, generation, mark_active=False)
        self.last_offspring_histories = set(offspring_mapping.values())
        self.last_offspring_smiles = set(offspring_mapping.keys())

        return str(offspring_docked_file), str(offspring_lineage_file) if offspring_lineage_file else None
    def _combine_lineage_records(
        self,
        generation: int,
        unique_smiles: List[str],
        crossover_entries: List[Dict],
        mutation_entries: List[Dict]
    ) -> List[Dict]:
        """合并交叉与突变的血统记录，并对齐到去重后的子代集合。"""
        lineage_map: Dict[str, List[Dict]] = {}
        for entry in crossover_entries + mutation_entries:
            child = entry.get("child")
            if not child:
                continue
            source_info = {
                "operation": entry.get("operation"),
                "parents": entry.get("parents", [])
            }
            if entry.get("operation") == "mutation":
                if "mutation_rule" in entry:
                    source_info["mutation_rule"] = entry["mutation_rule"]
                if "mutation_reaction_id" in entry:
                    source_info["mutation_reaction_id"] = entry["mutation_reaction_id"]
                if "complementary_molecules" in entry:
                    source_info["complementary_molecules"] = entry["complementary_molecules"]
            sources = lineage_map.setdefault(child, [])
            if source_info not in sources:
                sources.append(source_info)

        lineage_entries: List[Dict] = []
        for smi in unique_smiles:
            sources = lineage_map.get(smi)
            if not sources:
                continue
            lineage_entries.append({
                "generation": generation,
                "child": smi,
                "sources": sources
            })
        return lineage_entries
    def _save_next_generation_lineage(self, generation: int, next_parents_file: str, offspring_lineage_file: Optional[str]) -> None:
        """保存下一代父代的血统信息，便于追踪分子来源。"""
        if not self.enable_lineage_tracking:
            return
        if not next_parents_file:
            return
        gen_dir = self.output_dir / f"generation_{generation}"
        output_path = gen_dir / "next_generation_parents_lineage.jsonl"
        parents_smiles = self._read_smiles_from_file(Path(next_parents_file))
        offspring_entries = self._read_jsonl(Path(offspring_lineage_file)) if offspring_lineage_file else []
        offspring_map = {entry.get("child"): entry for entry in offspring_entries}

        records: List[Dict] = []
        for smi in parents_smiles:
            history = self.lineage_tracker.get(smi, [])
            latest_sources: List[Dict] = []
            origin = "unknown"
            if history:
                latest_event = history[-1]
                latest_sources = latest_event.get("sources", [])
                origin = "offspring" if latest_event.get("generation") == generation else "carryover"
            elif smi in offspring_map:
                latest_sources = offspring_map[smi].get("sources", [])
                origin = "offspring"
            records.append({
                "generation": generation,
                "child": smi,
                "origin": origin,
                "sources": latest_sources
            })

        self._write_jsonl(output_path, records)

    def run_selection(self, parent_docked_file: str, offspring_docked_file: str, generation: int) -> Optional[str]:
        """
        执行选择操作，从父代和子代中选出下一代。
        
        Args:
            parent_docked_file (str): 父代对接结果文件路径。
            offspring_docked_file (str): 子代对接结果文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[str]: 成功则返回下一代父代文件路径,失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始选择操作...")
        next_parents_file = self.output_dir / f"generation_{generation+1}" / "initial_population_docked.smi"
        next_parents_file.parent.mkdir(exist_ok=True)

        selection_config = self.config.get('selection', {})
        selection_mode = selection_config.get('selection_mode', 'single_objective')

        selection_succeeded = False
        if selection_mode == 'single_objective':
            logger.info("执行单目标选择...")
            selection_args = [
                '--docked_file', offspring_docked_file,
                '--parent_file', parent_docked_file,
                '--output_file', str(next_parents_file),
                '--config_file', self.config_path
            ]
            selection_succeeded = self._run_script('operations/selecting/molecular_selection.py', selection_args)
        
        elif selection_mode == 'multi_objective':
            logger.info("执行多目标选择...")
            multi_obj_config = selection_config.get('multi_objective_settings', {})
            n_select = multi_obj_config.get('n_select', 100)
            selection_args = [
                '--docked_file', offspring_docked_file,
                '--parent_file', parent_docked_file,
                '--output_file', str(next_parents_file),
                '--n_select', str(n_select),
                '--config_file', self.config_path,
                '--cache_file', str(self.metric_cache.cache_path),
                '--output_format', 'with_scores',
            ]
            selection_succeeded = self._run_script('operations/selecting/selecting_multi_demo.py', selection_args)
        
        else:
            logger.error(f"不支持的选择模式: {selection_mode}")
            return None

        if not selection_succeeded or self._count_molecules(str(next_parents_file)) == 0:
            logger.error(f"第 {generation} 代: 选择操作失败或未选出任何分子。")
            return None
        
        selected_count = self._count_molecules(str(next_parents_file))
        logger.info(f"选择操作完成 ({selection_mode}): 选出 {selected_count} 个分子作为下一代父代。")
        
        return str(next_parents_file)

    def run_selected_population_evaluation(self, selected_parents_file: str, generation: int) -> bool:
        """
        对选择后的精英种群（下一代父代）进行评分分析        
        Args:
            selected_parents_file (str): 选择后的下一代父代文件路径
            generation (int): 当前代数            
        Returns:
            bool: 评分分析是否成功
        """
        logger.info(f"第 {generation} 代: 开始对选择后的精英种群进行评分分析")
        
        gen_dir = self.output_dir / f"generation_{generation}"
        scoring_report_file = gen_dir / f"generation_{generation}_evaluation.txt"
        
        scoring_succeeded = self._run_script('operations/scoring/scoring_demo.py', [
            '--current_population_docked_file', str(selected_parents_file),
            '--initial_population_file', self.initial_population_file,
            '--output_file', str(scoring_report_file)
        ])        
        if scoring_succeeded:
            logger.info(f"第 {generation} 代: 精英种群评分分析完成，报告保存到 {scoring_report_file}")
        else:
            logger.warning(f"第 {generation} 代: 精英种群评分分析失败，但不影响主流程")            
        return scoring_succeeded

    def run_complete_workflow(self):
        """
        执行完整的FragEvo工作流。
        """
        logger.info(f"开始执行完整的FragEvo工作流程 (输出目录: {self.output_dir})")
        
        # 第0步：初代种群处理
        current_parents_docked_file = self.run_initial_generation()
        if not current_parents_docked_file:
            logger.error("初代种群处理失败，工作流终止。")
            return False
        
        logger.info(f"初代种群处理成功，结果文件: {current_parents_docked_file}")

        # 开始迭代
        for generation in range(1, self.max_generations + 1):
            next_parents_file = self.run_generation_step(generation, current_parents_docked_file)
            if not next_parents_file:
                logger.error(f"第 {generation} 代处理失败，工作流终止。")
                return False
            current_parents_docked_file = next_parents_file
        
        logger.info("=" * 60)
        logger.info("FragEvo工作流程全部完成!")
        logger.info(f"最终优化种群保存在: {current_parents_docked_file}")
        logger.info("=" * 60)
        self._export_evomo_files()
        
        return True

    def run_initial_generation(self) -> Optional[str]:
        """
        执行初代种群的处理，包括去重、格式化和对接。
        
        Returns:
            Optional[str]: 成功则返回初代种群对接结果文件的路径，失败则返回None。
        """
        logger.info("开始处理初代种群 (Generation 0)...")
        
        gen_dir = self.output_dir / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 检查初始种群文件是否存在
        if not Path(self.initial_population_file).exists():
            logger.error(f"初始种群文件未找到: {self.initial_population_file}")
            return None
        
        # 2. 去重并格式化初始种群
        initial_formatted_file = gen_dir / "initial_population_formatted.smi"
        unique_count = self._remove_duplicates_from_smiles_file(
            self.initial_population_file, 
            str(initial_formatted_file)
        )
        if unique_count == 0:
            logger.error("初始种群文件为空或处理失败。")
            return None

        # 记录初代种群血统信息
        self._record_initial_population(initial_formatted_file)
        
        # 3. 对初代种群进行对接
        initial_docked_file = gen_dir / "initial_population_docked.smi"
        
        # 显式传递处理器数量
        num_processors = self.config.get('performance', {}).get('number_of_processors')
        
        docking_args = [
            '--smiles_file', str(initial_formatted_file),
            '--output_file', str(initial_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path,
            '--seed', str(getattr(self, "seed", 42)),
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
            
        # 将处理器数量添加到命令行参数中
        if num_processors is not None:
            docking_args.extend(['--number_of_processors', str(num_processors)])

        docking_succeeded = self._run_script('operations/docking/docking_demo_finetune.py', docking_args)
        
        docked_count = self._count_molecules(str(initial_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("初代种群对接失败或未生成任何有效对接结果。")
            return None
        mapping = self._ingest_population_metrics(initial_docked_file, generation=0, mark_active=True)
        self._mark_histories_active(set(mapping.values()), generation=0)
        
        logger.info(f"初代种群对接完成: {docked_count} 个分子已评分。")
        return str(initial_docked_file)

    def run_generation_step(self, generation: int, current_parents_docked_file: str):
        """
        执行单代FragEvo的完整流程。
        """
        logger.info(f"========== 开始第 {generation} 代进化 ==========")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        # 1. 从父代对接文件中提取纯SMILES
        parent_smiles_file = gen_dir / "current_parent_smiles.smi"
        if not self._extract_smiles_from_docked_file(current_parents_docked_file, str(parent_smiles_file)):
            logger.error(f"第{generation}代: 无法从父代文件提取SMILES,工作流终止")
            return None

        # 2. 分解与掩码
        masked_file = self.run_decomposition_and_masking(str(parent_smiles_file), generation)
        if not masked_file:
            # 如果分解失败，可以决定是终止还是跳过GPT步骤
            logger.warning(f"第{generation}代: 分解掩码步骤失败,将跳过GPT生成。")
            gpt_generated_file = None
        else:
            # 3. GPT生成
            gpt_generated_file = self.run_gpt_generation(masked_file, generation)

        self._export_tokenized_representations(
            generation,
            parent_smiles_file,
            masked_file,
            gpt_generated_file
        )

        # 4. 遗传操作
        ga_children_files = self.run_ga_operations(str(parent_smiles_file), gpt_generated_file, generation)
        if not ga_children_files:
            logger.error(f"第{generation}代: 遗传操作失败，工作流终止。")
            return None
        
        crossover_file, mutation_file, crossover_lineage, mutation_lineage = ga_children_files

        # 5. 子代评估（对接，同时生成血统记录）
        offspring_result = self.run_offspring_evaluation(
            crossover_file,
            mutation_file,
            generation,
            crossover_lineage,
            mutation_lineage
        )
        if not offspring_result:
            logger.error(f"第{generation}代: 子代评估失败，工作流终止。")
            return None
        offspring_docked_file, offspring_lineage_file = offspring_result

        # 6. 选择
        next_parents_docked_file = self.run_selection(
            current_parents_docked_file, 
            offspring_docked_file, 
            generation
        )
        if not next_parents_docked_file:
            logger.error(f"第{generation}代: 选择操作失败，工作流终止。")
            return None

        selected_mapping = self._ingest_population_metrics(next_parents_docked_file, generation, mark_active=True)
        new_active_histories = set(selected_mapping.values())
        candidate_histories = set(self.current_active_histories)
        candidate_histories.update(self.last_offspring_histories)
        removed_histories = candidate_histories - new_active_histories
        if removed_histories:
            self._mark_histories_removed(removed_histories, generation)
        self._mark_histories_active(new_active_histories, generation)
        self.last_offspring_histories = set()
        self.last_offspring_smiles = set()

        
        # 保存下一代父代的血统信息，便于追踪
        if self.enable_lineage_tracking:
            self._save_next_generation_lineage(
                generation,
                next_parents_docked_file,
                offspring_lineage_file
            )

        # 7. 对选择后的精英种群进行评分分析（这是新的逻辑）
        self.run_selected_population_evaluation(next_parents_docked_file, generation)

        # 8. 清理临时文件（如果启用）
        self._cleanup_generation_files(generation)

        logger.info(f"========== 第 {generation} 代进化完成 ==========")
        return next_parents_docked_file

    def _cleanup_generation_files(self, generation_num: int):
        """
        清理指定代数产生的临时文件和目录
        
        Args:
            generation_num (int): 要清理的代数
        """
        if not self.config.get('performance', {}).get('cleanup_intermediate_files', False):
            return
            
        try:
            gen_dir = self.output_dir / f"generation_{generation_num}"
            if not gen_dir.exists():
                return
                
            # 清理对接过程中产生的临时文件夹
            temp_dirs_to_clean = [
                gen_dir / "ligands",
                gen_dir / "ligands3D_SDFs", 
                gen_dir / "ligands3D_PDBs",
                gen_dir / "docking_results" / "ligands",
                gen_dir / "docking_results" / "ligands3D_SDFs",
                gen_dir / "gpt_generated" / "docking_files"
            ]
            
            for temp_dir in temp_dirs_to_clean:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug(f"已清理临时目录: {temp_dir}")
            
            # 清理大型中间文件（但保留重要的结果文件）
            temp_files_to_clean = [
                gen_dir / "ga_input_pool.smi",
                gen_dir / "crossover_raw.smi", 
                gen_dir / "mutation_raw.smi",
                gen_dir / "offspring_combined_raw.smi",
                gen_dir / "offspring_formatted_for_docking.smi"
            ]
            
            for temp_file in temp_files_to_clean:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"已清理临时文件: {temp_file}")
                    
            logger.info(f"第 {generation_num} 代临时文件清理完成")
            
        except Exception as e:
            logger.warning(f"清理第 {generation_num} 代临时文件时出错: {e}")

# --- 主函数入口 ---
def main():
    """主函数，用于解析命令行参数和启动工作流"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FragEvo混合工作流执行器')
    parser.add_argument('--config', type=str, 
                       default='fragevo/config_example.json',
                       help='配置文件路径')
    parser.add_argument('--receptor', type=str, default=None,
                       help='(可选) 要运行的目标受体名称')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='(可选) 指定输出目录，覆盖配置文件中的设置')
    
    args = parser.parse_args()
    
    try:
        executor = FragEvoWorkflowExecutor(args.config, args.receptor, args.output_dir)
        success = executor.run_complete_workflow()
        if not success:
            logger.error("FragEvo工作流执行失败。")
            return 1
        
        logger.info("FragEvo工作流成功完成!")

    except Exception as e:
        logger.critical(f"工作流执行过程中发生严重错误: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
