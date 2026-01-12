#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置参数快照生成器
==================
该模块负责生成并保存当次GA运行的完整配置快照。
主要功能:
1. 根据实际执行模式过滤掉未使用的配置分支
2. 收集所有在运行中实际使用的参数
3. 生成一个"干净"且"完整"的参数记录文件


"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigSnapshotGenerator:
    """配置快照生成器类"""
    
    def __init__(self, original_config: Dict[str, Any], execution_context: Dict[str, Any]):
        """
        初始化配置快照生成器
        
        Args:
            original_config: 原始完整配置字典
            execution_context: 执行上下文信息，包含实际使用的模式和参数
        """
        self.original_config = original_config
        self.execution_context = execution_context
        self.used_config = {}
        
    def generate_snapshot(self) -> Dict[str, Any]:
        """
        生成配置快照
        
        Returns:
            Dict: 包含当次运行实际使用参数的配置字典
        """
        logger.info("开始生成配置参数快照...")
        
        # 1. 添加基本执行信息
        self._add_execution_metadata()
        
        # 2. 处理对接相关参数
        self._process_docking_config()
        
        # 3. 处理受体配置
        self._process_receptor_config()
        
        # 4. 处理选择策略配置（核心：根据实际模式过滤）
        self._process_selection_config()
        
        # 5. 处理交叉操作参数
        self._process_crossover_config()
        
        # 6. 处理突变操作参数
        self._process_mutation_config()
        
        # 7. 处理过滤参数
        self._process_filter_config()
        
        # 8. 处理GPT相关参数
        self._process_gpt_config()
        
        # 9. 处理工作流参数
        self._process_workflow_config()
        
        logger.info("配置参数快照生成完成")
        return self.used_config
    
    def _add_execution_metadata(self):
        """添加执行元数据"""
        self.used_config["execution_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "config_file_path": self.execution_context.get("config_file_path"),
            "project_root": self.execution_context.get("project_root"),
            "base_output_dir": self.execution_context.get("base_output_dir"),
            "receptor_name": self.execution_context.get("receptor_name"),
            "run_specific_output_dir": self.execution_context.get("run_specific_output_dir"),
            "max_generations": self.execution_context.get("max_generations"),
            "initial_population_file": self.execution_context.get("initial_population_file")
        }
    
    def _process_docking_config(self):
        """处理对接相关参数"""
        docking_config = self.original_config.get("docking", {})
        if docking_config:
            self.used_config["docking"] = {
                "dock_choice": docking_config.get("dock_choice"),
                "conversion_choice": docking_config.get("conversion_choice"),
                "docking_exhaustiveness": docking_config.get("docking_exhaustiveness"),
                "docking_num_modes": docking_config.get("docking_num_modes"),
                "seed": docking_config.get("seed"),
                "number_of_processors": docking_config.get("number_of_processors"),
                "max_variants_per_compound": docking_config.get("max_variants_per_compound"),
                "gypsum_thoroughness": docking_config.get("gypsum_thoroughness"),
                "gypsum_timeout_limit": docking_config.get("gypsum_timeout_limit"),
                "min_ph": docking_config.get("min_ph"),
                "max_ph": docking_config.get("max_ph"),
                "pka_precision": docking_config.get("pka_precision"),
                "debug_mode": docking_config.get("debug_mode"),
                "mgltools_dir": docking_config.get("mgltools_dir"),
                "mgl_python": docking_config.get("mgl_python"),
                "prepare_receptor4.py": docking_config.get("prepare_receptor4.py"),
                "prepare_ligand4.py": docking_config.get("prepare_ligand4.py"),
                "docking_executable": docking_config.get("docking_executable"),
                "timeout_vs_gtimeout": docking_config.get("timeout_vs_gtimeout")
            }
    
    def _process_receptor_config(self):
        """处理受体配置"""
        receptors_config = self.original_config.get("receptors", {})
        if receptors_config:
            # 只保存实际使用的受体信息
            used_receptor_config = {
                "description": "当次运行实际使用的受体配置"
            }
            
            receptor_name = self.execution_context.get("receptor_name")
            if receptor_name and receptor_name != "default_receptor":
                # 使用了特定受体
                target_list = receptors_config.get("target_list", {})
                if receptor_name in target_list:
                    used_receptor_config["used_receptor"] = {
                        "name": receptor_name,
                        **target_list[receptor_name]
                    }
            else:
                # 使用了默认受体
                default_receptor = receptors_config.get("default_receptor")
                if default_receptor:
                    used_receptor_config["used_receptor"] = {
                        "name": "default_receptor",
                        **default_receptor
                    }
            
            self.used_config["receptors"] = used_receptor_config
    
    def _process_selection_config(self):
        """处理选择策略配置（核心功能：根据实际模式过滤）"""
        selection_config = self.original_config.get("selection", {})
        if not selection_config:
            return
            
        # 获取实际使用的选择模式
        actual_selection_mode = self.execution_context.get("selection_mode", 
                                                           selection_config.get("selection_mode", "single_objective"))
        
        # 构建干净的选择配置
        clean_selection_config = {
            "description": "当次运行实际使用的选择策略配置",
            "selection_mode": actual_selection_mode
        }
        
        # 根据实际模式只保留相关的配置
        if actual_selection_mode == "single_objective":
            single_settings = selection_config.get("single_objective_settings", {})
            clean_selection_config["single_objective_settings"] = {
                "description": single_settings.get("description"),
                "n_select": single_settings.get("n_select"),
                "selector_choice": single_settings.get("selector_choice"),
                "tourn_size": single_settings.get("tourn_size"),
                "enable_dynamic_selection": single_settings.get("enable_dynamic_selection"),
                "dynamic_selection_transition_generation": single_settings.get("dynamic_selection_transition_generation"),
                "early_stage_selector": single_settings.get("early_stage_selector"),
                "late_stage_selector": single_settings.get("late_stage_selector")
            }
        elif actual_selection_mode == "multi_objective":
            multi_settings = selection_config.get("multi_objective_settings", {})
            clean_selection_config["multi_objective_settings"] = {
                "description": multi_settings.get("description"),
                "n_select": multi_settings.get("n_select"),
                "objectives": multi_settings.get("objectives"),
                "enable_crowding_distance": multi_settings.get("enable_crowding_distance"),
                "verbose": multi_settings.get("verbose")
            }
        
        self.used_config["selection"] = clean_selection_config
    
    def _process_crossover_config(self):
        """处理交叉操作参数"""
        crossover_config = self.original_config.get("crossover_finetune", {})
        if crossover_config:
            number_of_crossovers = crossover_config.get(
                "number_of_crossovers",
                crossover_config.get("crossover_attempts"),  # backward-compat
            )
            self.used_config["crossover_finetune"] = {
                "number_of_crossovers": number_of_crossovers,
                "min_atom_match_mcs": crossover_config.get("min_atom_match_mcs"),
                "max_time_mcs_prescreen": crossover_config.get("max_time_mcs_prescreen"),
                "max_time_mcs_thorough": crossover_config.get("max_time_mcs_thorough"),
                "protanate_step": crossover_config.get("protanate_step")
            }
    
    def _process_mutation_config(self):
        """处理突变操作参数"""
        mutation_config = self.original_config.get("mutation_finetune", {})
        if mutation_config:
            number_of_mutants = mutation_config.get(
                "number_of_mutants",
                mutation_config.get("mutation_attempts"),  # backward-compat
            )
            self.used_config["mutation_finetune"] = {
                "number_of_mutants": number_of_mutants,
                "rxn_library": mutation_config.get("rxn_library"),
                "rxn_library_file": mutation_config.get("rxn_library_file"),
                "function_group_library": mutation_config.get("function_group_library"),
                "complementary_mol_directory": mutation_config.get("complementary_mol_directory")
            }
    
    def _process_filter_config(self):
        """处理过滤参数"""
        filter_config = self.original_config.get("filter", {})
        if filter_config:
            self.used_config["filter"] = {
                "enable_lipinski_strict": filter_config.get("enable_lipinski_strict"),
                "enable_lipinski_lenient": filter_config.get("enable_lipinski_lenient"),
                "enable_ghose": filter_config.get("enable_ghose"),
                "enable_ghose_modified": filter_config.get("enable_ghose_modified"),
                "enable_mozziconacci": filter_config.get("enable_mozziconacci"),
                "enable_vande_waterbeemd": filter_config.get("enable_vande_waterbeemd"),
                "enable_pains": filter_config.get("enable_pains"),
                "enable_nih": filter_config.get("enable_nih"),
                "enable_brenk": filter_config.get("enable_brenk"),
                "no_filters": filter_config.get("no_filters"),
                "alternative_filters": filter_config.get("alternative_filters"),
                "target_diversity": filter_config.get("target_diversity"),
                "target_exploitation": filter_config.get("target_exploitation")
            }
    
    def _process_gpt_config(self):
        """处理GPT相关参数"""
        gpt_config = self.original_config.get("gpt", {})
        if gpt_config:
            self.used_config["gpt"] = {
                "mask_last_n_fragments": gpt_config.get("mask_last_n_fragments"),
                "model_path": gpt_config.get("model_path"),
                "temperature": gpt_config.get("temperature"),
                "max_length": gpt_config.get("max_length"),
                "num_generated_per_input": gpt_config.get("num_generated_per_input")
            }
    
    def _process_workflow_config(self):
        """处理工作流参数"""
        workflow_config = self.original_config.get("workflow", {})
        if workflow_config:
            self.used_config["workflow"] = {
                "seed": workflow_config.get("seed"),
                "initial_population_file": workflow_config.get("initial_population_file"),
                "output_directory": workflow_config.get("output_directory"),
                "max_generations": workflow_config.get("max_generations"),
                "early_stopping_patience": workflow_config.get("early_stopping_patience"),
                "log_level": workflow_config.get("log_level"),
                "save_intermediate_results": workflow_config.get("save_intermediate_results"),
                "enable_adaptive_adjustment": workflow_config.get("enable_adaptive_adjustment")
            }

def save_config_snapshot(original_config: Dict[str, Any], 
                        execution_context: Dict[str, Any], 
                        output_file_path: str) -> bool:
    """
    保存配置参数快照到文件
    
    Args:
        original_config: 原始完整配置字典
        execution_context: 执行上下文信息
        output_file_path: 输出文件路径
        
    Returns:
        bool: 是否成功保存
    """
    try:
        # 生成配置快照
        generator = ConfigSnapshotGenerator(original_config, execution_context)
        snapshot = generator.generate_snapshot()
        
        # 确保输出目录存在
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存到文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置参数快照已保存到: {output_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存配置参数快照失败: {e}", exc_info=True)
        return False

def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='配置参数快照生成器')
    parser.add_argument('--config', type=str, required=True, help='--config')
    parser.add_argument('--output', type=str, required=True, help='--output')
    parser.add_argument('--selection_mode', type=str, default='multi_objective', 
                       help='--selection_mode')
    
    args = parser.parse_args()
    
    # 加载原始配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 构建执行上下文（测试用）
    execution_context = {
        "config_file_path": args.config,
        "selection_mode": args.selection_mode,
        "receptor_name": "default_receptor",
        "max_generations": config.get("workflow", {}).get("max_generations", 5)
    }
    
    # 生成并保存快照
    success = save_config_snapshot(config, execution_context, args.output)
    
    if success:
        print(f"配置快照生成成功: {args.output}")
    else:
        print("配置快照生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 
