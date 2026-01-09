#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA操作执行脚本
==============
集成所有GA操作的完整遗传算法工作流程。
实现：初代种群 -> 迭代(交叉/突变 -> 评估 -> 选择) -> 优化分子生成
"""
import os
import sys
import json
import subprocess
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from operations.stating.config_snapshot_generator import save_config_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
class GAWorkflowExecutor:       
    def __init__(self, config_path: str, receptor_name: Optional[str] = None, output_dir_override: Optional[str] = None):        
        self.config_path = config_path
        self.config = self._load_config()        
        self.run_params = {}        
        self._setup_parameters_and_paths(receptor_name, output_dir_override)        
        self._save_run_parameters()        
        logger.info(f"GA工作流初始化完成, 输出目录: {self.output_dir}")
        logger.info(f"最大迭代代数: {self.max_generations}")
        
    def _load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)         
    def _setup_parameters_and_paths(self, receptor_name: Optional[str], output_dir_override: Optional[str]):        
        self.project_root = Path(self.config.get('paths', {}).get('project_root', PROJECT_ROOT))
        workflow_config = self.config.get('workflow', {})
        # 1. 记录使用的配置文件和项目根目录
        self.run_params['config_file_path'] = self.config_path
        self.run_params['project_root'] = str(self.project_root)
        # 2. 确定并记录输出目录
        if output_dir_override:
            output_dir_name = output_dir_override
            logger.info(f"使用命令行指定的输出目录: {output_dir_name}")
        else:
            output_dir_name = workflow_config.get('output_directory', 'GA_output')
        base_output_dir = self.project_root / output_dir_name
        self.run_params['base_output_dir'] = str(base_output_dir)
        # 3. 确定并记录受体信息
        self.receptor_name = receptor_name
        if self.receptor_name:
            self.output_dir = base_output_dir / self.receptor_name
            logger.info(f"将为受体 '{self.receptor_name}' 在指定目录中创建输出。")
            self.run_params['receptor_name'] = self.receptor_name
        else:
            self.output_dir = base_output_dir / "default_receptor_run"
            logger.info("未指定受体，将在默认目录中创建输出。")
            self.run_params['receptor_name'] = "default_receptor"        
        self.run_params['run_specific_output_dir'] = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 4. 记录其他GA核心参数
        self.max_generations = workflow_config.get('max_generations', 5)
        self.initial_population_file = workflow_config.get('initial_population_file', 'datasets/source_compounds/naphthalene_smiles.smi')
        self.run_params['max_generations'] = self.max_generations
        self.run_params['initial_population_file'] = self.initial_population_file        
        # 5. 记录实际使用的选择模式
        selection_config = self.config.get('selection', {})
        self.run_params['selection_mode'] = selection_config.get('selection_mode', 'single_objective')

    def _save_run_parameters(self):        
        execution_context = {
            "config_file_path": self.run_params.get('config_file_path'),
            "project_root": self.run_params.get('project_root'),
            "base_output_dir": self.run_params.get('base_output_dir'),
            "receptor_name": self.run_params.get('receptor_name'),
            "run_specific_output_dir": self.run_params.get('run_specific_output_dir'),
            "max_generations": self.run_params.get('max_generations'),
            "initial_population_file": self.run_params.get('initial_population_file'),
            "selection_mode": self.run_params.get('selection_mode')
        }                
        snapshot_file_path = self.output_dir / "execution_config_snapshot.json"
        success = save_config_snapshot(
            original_config=self.config,
            execution_context=execution_context,
            output_file_path=str(snapshot_file_path)
        )        
        if success:
            logger.info(f"完整的执行配置快照已保存到: {snapshot_file_path}")
        else:
            logger.error("保存执行配置快照失败")            
        # 为了向后兼容，也保留原来的简化版本
        legacy_vars_path = self.output_dir / "vars.json"
        with open(legacy_vars_path, 'w', encoding='utf-8') as f:
            json.dump(self.run_params, f, indent=4, ensure_ascii=False)
        logger.info(f"简化版运行参数已保存到: {legacy_vars_path}")
    
    def _run_script(self, script_path: str, args: List[str]) -> bool:
        """运行Python脚本,并管理输出信息"""
        full_script_path = self.project_root / script_path
        cmd = ['python', str(full_script_path)] + args        
        # 将详细命令的日志级别降为DEBUG
        logger.debug(f"执行命令: {' '.join(cmd)}")        
        # 捕获输出，但在成功时不显示stdout/stderr，以保持日志整洁
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root), check=False)
        if result.returncode == 0:
            logger.info(f"脚本 {script_path} 执行成功")
            return True
        else:
            logger.error(f"脚本 {script_path} 执行失败")
            # 仅在失败时打印详细的错误输出
            logger.error(f"错误输出 (stderr):\n{result.stderr}")
            if result.stdout:
                logger.error(f"标准输出 (stdout):\n{result.stdout}")
            return False
    
    def _count_molecules(self, file_path: str) -> int:
        """统计文件中分子数量"""
        with open(file_path, 'r') as f:
            count = sum(1 for line in f if line.strip())
        return count
    
    def _remove_duplicates_from_smiles_file(self, input_file: str, output_file: str) -> int:
        """
        去除SMILES文件中的重复分子,并为每个分子添加唯一ID。
        输出格式: SMILES  ligand_id_X
        """
        try:
            unique_smiles = set()
            with open(input_file, 'r') as f:
                for line in f:
                    smiles = line.strip().split()[0]  # 取第一列作为SMILES
                    if smiles:
                        unique_smiles.add(smiles)
            
            # 转换为列表以保证顺序
            unique_smiles_list = sorted(list(unique_smiles))
            
            with open(output_file, 'w') as f:
                for i, smiles in enumerate(unique_smiles_list):
                    ligand_id = f"ligand_id_{i}"
                    f.write(f"{smiles}\t{ligand_id}\n")
            
            logger.info(f"去重完成: {len(unique_smiles_list)} 个独特分子保存到 {output_file} (已添加ID)")
            return len(unique_smiles_list)
    
    def _combine_files(self, file_list: List[str], output_file: str) -> bool:
        """合并多个文件"""
        try:
            with open(output_file, 'w') as outf:
                for file_path in file_list:
                    with open(file_path, 'r') as inf:
                        for line in inf:
                            line = line.strip()
                            if line:
                                outf.write(line + '\n')
            return True
    
    def _extract_smiles_from_docked_file(self, docked_file: str, output_smiles_file: str) -> bool:#提取分数       
        try:
            with open(docked_file, 'r') as infile, open(output_smiles_file, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if line:
                        # 提取第一列作为SMILES
                        smiles = line.split()[0]
                        outfile.write(f"{smiles}\n")
            
            extracted_count = self._count_molecules(output_smiles_file)
            logger.debug(f"从 {docked_file} 提取了 {extracted_count} 个SMILES到 {output_smiles_file}")
            return True
    
    def run_initial_generation(self) -> str:
        """
        执行初代种群处理        
        Returns:
            初代种群对接结果文件路径
        """
        logger.info("开始处理初代种群 (Generation 0)...")        
        gen_dir = self.output_dir / "generation_0"
        gen_dir.mkdir(exist_ok=True)        
        # 1. 去重初始种群
        initial_unique_file = gen_dir / "initial_population_unique.smi"
        unique_count = self._remove_duplicates_from_smiles_file(
            self.initial_population_file, 
            str(initial_unique_file)
        )        
        if unique_count == 0:
            logger.error("初始种群去重失败")
            return None        
        # 2. 对初代种群进行对接
        initial_docked_file = gen_dir / "initial_population_docked.smi"
        docking_args = [
            '--smiles_file', str(initial_unique_file),
            '--output_file', str(initial_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])            
        docking_succeeded = self._run_script('operations/docking/docking_demo_finetune.py', docking_args)        
        docked_count = self._count_molecules(str(initial_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("初代种群对接失败或未生成任何有效对接结果。请检查对接模块配置和日志。")
            return None        
        logger.info(f"初代种群对接完成: {docked_count} 个分子")        
        return str(initial_docked_file)
    
    def run_genetic_operations(self, parent_file: str, generation: int) -> tuple:
        """
        执行遗传操作：交叉和突变        
        Args:
            parent_file: 父代分子文件路径
            generation: 当前代数            
        Returns:
            (crossover_file, mutation_file): 交叉和突变结果文件路径
        """
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)        
        # 1. 交叉操作
        crossover_raw_file = gen_dir / "crossover_raw.smi"#交叉原始文件（生成）
        crossover_filtered_file = gen_dir / "crossover_filtered.smi"        
        crossover_succeeded = self._run_script('operations/crossover/crossover_demo_finetune.py', [
            '--smiles_file', parent_file,
            '--output_file', str(crossover_raw_file)
        ])        
        # 过滤交叉结果
        filter_crossover_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(crossover_raw_file),
            '--output_file', str(crossover_filtered_file)
        ])
        
        crossover_count = self._count_molecules(str(crossover_filtered_file))
        if not crossover_succeeded or not filter_crossover_succeeded:
            logger.error("交叉或过滤步骤执行失败。")
            return None, None        
        logger.info(f"交叉操作完成: 生成 {crossover_count} 个新分子 (过滤后)")
        
        # 2. 突变操作
        mutation_raw_file = gen_dir / "mutation_raw.smi"#突变原始文件（生成）
        mutation_filtered_file = gen_dir / "mutation_filtered.smi"
        
        mutation_succeeded = self._run_script('operations/mutation/mutation_demo_finetune.py', [
            '--smiles_file', parent_file,
            '--output_file', str(mutation_raw_file)
        ])        
        # 过滤突变结果
        filter_mutation_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(mutation_raw_file),
            '--output_file', str(mutation_filtered_file)
        ])
        
        mutation_count = self._count_molecules(str(mutation_filtered_file))
        if not mutation_succeeded or not filter_mutation_succeeded:
            logger.error("突变或过滤步骤执行失败。")
            return None, None
        
        logger.info(f"突变操作完成: 生成 {mutation_count} 个新分子 (过滤后)")
        
        return str(crossover_filtered_file), str(mutation_filtered_file)
    
    def run_offspring_evaluation(self, crossover_file: str, mutation_file: str, generation: int) -> str:
        """
        执行子代评估:合并、去重、添加ID、对接        
        Args:
            crossover_file: 交叉结果文件
            mutation_file: 突变结果文件
            generation: 当前代数            
        Returns:
            子代对接结果文件路径
        """
        logger.info(f"开始第{generation}代子代评估")        
        gen_dir = self.output_dir / f"generation_{generation}"        
        # 1. 合并交叉和突变结果到一个临时文件
        offspring_raw_file = gen_dir / "offspring_combined_raw.smi"
        if not self._combine_files([crossover_file, mutation_file], str(offspring_raw_file)):
            logger.error("子代合并失败")
            return None        
        # 2. 对合并后的文件进行去重并添加唯一ID，确保格式正确
        offspring_formatted_file = gen_dir / "offspring_formatted_for_docking.smi"
        offspring_count = self._remove_duplicates_from_smiles_file(
            str(offspring_raw_file), 
            str(offspring_formatted_file)
        )
        if offspring_count == 0:
            logger.error("子代分子在去重和格式化后为空。")
            return None
            
        logger.info(f"子代合并和格式化完成: 共 {offspring_count} 个独特的分子准备进行对接")        
        # 3. 对格式化后的子代文件进行对接
        offspring_docked_file = gen_dir / "offspring_docked.smi"
        docking_args = [
            '--smiles_file', str(offspring_formatted_file),
            '--output_file', str(offspring_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])            
        docking_succeeded = self._run_script('operations/docking/docking_demo_finetune.py', docking_args)
        
        docked_count = self._count_molecules(str(offspring_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("子代对接失败或未生成任何有效对接结果。请检查对接模块配置和日志。")
            return None        
        logger.info(f"子代对接完成: {docked_count} 个分子")
        
        return str(offspring_docked_file)
    
    def run_selection(self, parent_docked_file: str, offspring_docked_file: str, generation: int) -> str:
        """
        执行选择操作：从父代+子代中选择下一代父代        
        Args:
            parent_docked_file: 父代对接结果文件
            offspring_docked_file: 子代对接结果文件
            generation: 当前代数            
        Returns:
            下一代父代文件路径
        """
        logger.info(f"开始第{generation}代选择操作")        
        gen_dir = self.output_dir / f"generation_{generation}"
        next_parents_file = gen_dir / f"next_generation_parents.smi"   
        
        selection_config = self.config.get('selection', {})
        selection_mode = selection_config.get('selection_mode', 'single_objective')        
        if selection_mode == 'single_objective':
            # 单目标选择：使用molecular_selection.py           
            single_obj_config = selection_config.get('single_objective_settings', {})            
            selection_args = [
                '--docked_file', offspring_docked_file,
                '--parent_file', parent_docked_file,
                '--output_file', str(next_parents_file),
                # 传递config_path，让选择脚本可以自己读取详细参数
                '--config_file', self.config_path 
            ]            
            selection_succeeded = self._run_script('operations/selecting/molecular_selection.py', selection_args)            
        elif selection_mode == 'multi_objective':
            # 多目标选择：使用selecting_multi_demo.py            
            multi_obj_config = selection_config.get('multi_objective_settings', {})
            n_select = multi_obj_config.get('n_select', 100)            
            selection_args = [
                '--docked_file', offspring_docked_file,
                '--parent_file', parent_docked_file,
                '--output_file', str(next_parents_file),
                '--n_select', str(n_select),
                '--output_format', 'with_scores'  # 确保输出格式包含分数
            ]            
            if multi_obj_config.get('verbose', False):
                selection_args.append('--verbose')            
            selection_succeeded = self._run_script('operations/selecting/selecting_multi_demo.py', selection_args)            
        else:
            logger.error(f"不支持的选择模式: {selection_mode}")
            return None        
        selected_count = self._count_molecules(str(next_parents_file))
        if not selection_succeeded or selected_count == 0:
            logger.error("选择操作失败或未选出任何分子。")
            return None        
        logger.info(f"选择操作完成 ({selection_mode}): 选出 {selected_count} 个下一代父代")        
        return str(next_parents_file)
    
    def run_selected_population_evaluation(self, selected_parents_file: str, generation: int) -> bool:
        """
        对选择后的精英种群（下一代父代）进行评分分析        
        Args:
            selected_parents_file: 选择后的下一代父代文件路径
            generation: 当前代数            
        Returns:
            bool: 评分分析是否成功
        """
        logger.info(f"开始对第{generation}代选择后的精英种群进行评分分析")        
        gen_dir = self.output_dir / f"generation_{generation}"
        scoring_report_file = gen_dir / f"generation_{generation}_evaluation.txt"
        
        scoring_succeeded = self._run_script('operations/scoring/scoring_demo.py', [
            '--current_population_docked_file', str(selected_parents_file),
            '--initial_population_file', self.initial_population_file,
            '--output_file', str(scoring_report_file)
        ])        
        if scoring_succeeded:
            logger.info(f"第{generation}代精英种群评分分析完成: 报告保存到 {scoring_report_file}")
        else:
            logger.warning(f"第{generation}代精英种群评分分析失败")            
        return scoring_succeeded
    def run_complete_workflow(self):
        """执行完整的GA工作流"""
        logger.info(f"开始执行完整的GA工作流程 (输出目录: {self.output_dir})")        
        # 第0步：初代种群处理
        current_parents_docked_file = self.run_initial_generation()
        if not current_parents_docked_file:
            logger.error("初代种群处理失败，工作流终止")
            return False        
        # 开始迭代
        for generation in range(1, self.max_generations + 1):
            logger.info(f"----- 开始第 {generation} 代进化 -----")            
            # 1. 从带分数的父代文件中提取纯SMILES用于遗传操作
            gen_dir = self.output_dir / f"generation_{generation}"
            gen_dir.mkdir(exist_ok=True)
            parent_smiles_file = gen_dir / "selected_parent_smiles.smi"#从带分数的父代文件中提取纯SMILES用于遗传操作            
            if not self._extract_smiles_from_docked_file(current_parents_docked_file, str(parent_smiles_file)):
                logger.error(f"第{generation}代: 无法从父代文件提取SMILES,工作流终止")
                return False            
            # 2. 遗传操作 (使用纯SMILES文件)
            crossover_file, mutation_file = self.run_genetic_operations(str(parent_smiles_file), generation)
            if not crossover_file or not mutation_file:
                logger.error(f"第{generation}代遗传操作失败，工作流终止")
                return False            
            # 3. 子代评估 (得到带分数的子代文件，但不进行评分分析)
            offspring_docked_file = self.run_offspring_evaluation(crossover_file, mutation_file, generation)
            # if not offspring_docked_file:
            #     logger.error(f"第{generation}代子代评估失败，工作流终止")
            #     return False            
            # 4. 选择操作：从父代(带分数) + 子代(带分数)中选择下一代父代(带分数)
            next_parents_docked_file = self.run_selection(current_parents_docked_file, offspring_docked_file, generation)
            if not next_parents_docked_file:
                logger.error(f"第{generation}代选择操作失败，工作流终止")
                return False            
            # 5. 对选择后的精英种群进行评分分析
            self.run_selected_population_evaluation(next_parents_docked_file, generation)            
            # 更新父代文件，准备下一代 (现在是带分数的文件)
            current_parents_docked_file = next_parents_docked_file            
            logger.info(f"第 {generation} 代进化完成")        
        logger.info("=" * 60)
        logger.info("GA工作流程全部完成!")
        logger.info(f"最终优化种群保存在: {current_parents_docked_file}")
        logger.info("=" * 60)        
        return True
def main():    
    import argparse    
    parser = argparse.ArgumentParser(description='GA完整工作流执行器 (可独立运行)')
    parser.add_argument('--config', type=str, default='fragevo/config_example.json')
    parser.add_argument('--receptor', type=str, default=None, help='(可选) 要运行的目标受体名称')
    parser.add_argument('--output_dir', type=str, default=None, help='(可选) 指定输出目录，覆盖配置文件中的设置')    
    args = parser.parse_args()    
    # 创建并运行GA工作流
    executor = GAWorkflowExecutor(args.config, args.receptor, args.output_dir)
    # 执行完整工作流
    success = executor.run_complete_workflow()
    if success:
        logger.info("GA工作流执行成功完成")
        return 0
    else:
        logger.error("GA工作流执行失败")
        return 1
if __name__ == "__main__":
    exit(main())
