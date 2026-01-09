#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA优化流程主入口(纯GA版本)
================
该脚本是整个GA优化实验的顶层控制器。
它负责解析命令行参数（如配置文件和目标受体），
然后调用核心工作流执行器来完成针对单个或多个受体的完整GA流程。
通过修改配置文件中的 'selection_mode'，可以切换单目标或多目标优化。
用法:
  - 针对默认受体运行 (使用配置文件中的默认模式):
    python fragevo/GA_main.py --config fragevo/config_example.json
  - 针对特定受体运行:
    python fragevo/GA_main.py --config fragevo/config_example.json --receptor 4r6e
  - 为所有受体运行:
    python fragevo/GA_main.py --config fragevo/config_example.json --all_receptors
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 将项目根目录添加到Python路径
# 假设fragevo/GA_main.py位于项目根目录下的fragevo文件夹中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 从重构后的执行器模块中导入核心类
from operations.operations_execute_GA import GAWorkflowExecutor
def main():
    """主函数:解析参数并启动GA工作流"""
    parser = argparse.ArgumentParser(description='GA分子优化流程主入口')
    parser.add_argument('--config', type=str, default='fragevo/config_example.json')
    parser.add_argument('--receptor', type=str, default=None,help='(可选) 指定要运行的目标受体名称。如果未提供，将使用默认受体。')
    parser.add_argument('--all_receptors', action='store_true',help='(可选) 运行配置文件中`target_list`的所有受体。如果使用此选项，将忽略--receptor参数。')
    parser.add_argument('--output_dir', type=str, default=None)   #在此设置的话就会覆盖掉参数配置json文件中的设置
    args = parser.parse_args()
    
    # --- 参数验证和准备 ---
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    receptors_to_run = []
    if args.all_receptors:
        logger.info("检测到 --all_receptors 标志，将为配置文件中的所有受体运行工作流。")
        # 确保 target_list 存在
        target_list = config.get('receptors', {}).get('target_list', {})
        if not target_list:
            logger.warning("配置文件中未找到 'receptors.target_list' 或列表为空，无受体可运行。")
        else:
            receptors_to_run = list(target_list.keys())
        
        logger.info(f"计划运行的受体列表: {receptors_to_run}")
    else:
        # 如果未指定 --all_receptors，则运行单个受体（可以是指定的或默认的）
        receptors_to_run.append(args.receptor)

    # --- 循环启动工作流 ---
    if not receptors_to_run:
        logger.info("没有需要运行的受体，程序退出。")
        sys.exit(0)
        
    successful_runs = []
    failed_runs = []
    for receptor_name in receptors_to_run:
        receptor_display_name = receptor_name or '默认受体'
        try:
            logger.info("=" * 80)
            logger.info(f"开始为受体 '{receptor_display_name}' 运行GA工作流")
            logger.info(f"使用配置文件: {config_path}")
            logger.info("=" * 80)
            
            # 实例化工作流执行器，传入配置路径、受体名称和输出目录覆盖
            executor = GAWorkflowExecutor(
                config_path=str(config_path), 
                receptor_name=receptor_name,
                output_dir_override=args.output_dir
            )
            
            # 运行完整的GA流程
            success = executor.run_complete_workflow()
            
            if success:
                logger.info("=" * 60)
                logger.info(f"针对受体 '{receptor_display_name}' 的GA工作流成功完成!")
                logger.info("=" * 60)
                successful_runs.append(receptor_display_name)
            else:
                logger.error("=" * 60)
                logger.error(f"针对受体 '{receptor_display_name}' 的GA工作流执行失败。")
                logger.error("=" * 60)
                failed_runs.append(receptor_display_name)
                
        except Exception as e:
            logger.error(f"为受体 '{receptor_display_name}' 运行主流程时发生未处理的异常: {e}", exc_info=True)
            failed_runs.append(receptor_display_name)

    # --- 最终总结 ---
    logger.info("=" * 80)
    logger.info("所有GA工作流执行完毕。")
    if successful_runs:
        logger.info(f"成功运行的受体 ({len(successful_runs)}): {successful_runs}")
    if failed_runs:
        logger.error(f"失败的受体 ({len(failed_runs)}): {failed_runs}")
    logger.info("=" * 80)
    
    if failed_runs:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 
