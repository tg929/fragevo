#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU资源动态检测工具
==================
基于psutil库实现实时CPU空闲资源检测，支持智能并行资源分配
"""
import psutil
import time
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_available_cpu_cores(sample_duration: float = 1.0, cpu_threshold: float = 80.0) -> Tuple[int, float]:
    """
    动态检测当前系统可用的CPU核心数
    
    Args:
        sample_duration: CPU使用率采样持续时间（秒）
        cpu_threshold: CPU使用率阈值，超过此值的核心被认为"忙碌"
        
    Returns:
        Tuple[int, float]: (可用核心数, 当前系统平均CPU使用率)
    """
    try:
        # 获取系统总核心数
        total_cores = psutil.cpu_count(logical=True)
        
        # 采样CPU使用率 - 按核心分别统计
        cpu_percentages = psutil.cpu_percent(interval=sample_duration, percpu=True)
        
        # 计算系统平均使用率
        avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
        
        # 统计空闲核心数（使用率低于阈值的核心）
        available_cores = sum(1 for usage in cpu_percentages if usage < cpu_threshold)
        
        # 至少保证有1个核心可用
        available_cores = max(1, available_cores)
        
        logger.info(f"CPU资源检测结果:")
        logger.info(f"  - 系统总核心: {total_cores}")
        logger.info(f"  - 平均CPU使用率: {avg_cpu_usage:.1f}%")
        logger.info(f"  - 可用核心数 (使用率<{cpu_threshold}%): {available_cores}")
        
        return available_cores, avg_cpu_usage
        
    except Exception as e:
        logger.warning(f"CPU检测失败，使用默认值: {e}")
        # 降级方案：返回总核心数的80%
        fallback_cores = max(1, int(psutil.cpu_count() * 0.8))
        return fallback_cores, 0.0

def calculate_optimal_workers(target_count: int, available_cores: int, cores_per_worker: int) -> Tuple[int, int]:
    """
    计算最优的工作进程数和每进程核心数
    
    Args:
        target_count: 目标任务数（如受体数量）
        available_cores: 可用CPU核心数
        cores_per_worker: 每个工作进程期望的核心数
        
    Returns:
        Tuple[int, int]: (实际工作进程数, 每进程实际核心数)
    """
    if cores_per_worker == -1:
        # 自动模式：平均分配所有可用核心
        if target_count <= available_cores:
            # 核心充足，每个任务分配多个核心
            actual_workers = target_count
            actual_cores_per_worker = max(1, available_cores // target_count)
        else:
            # 核心不足，每个核心处理一个任务
            actual_workers = available_cores
            actual_cores_per_worker = 1
    else:
        # 固定模式：使用指定的核心数
        max_possible_workers = available_cores // cores_per_worker
        actual_workers = min(target_count, max_possible_workers, available_cores)
        actual_cores_per_worker = cores_per_worker
    
    # 保证至少有1个工作进程
    actual_workers = max(1, actual_workers)
    actual_cores_per_worker = max(1, actual_cores_per_worker)
    
    logger.info(f"并行资源分配结果:")
    logger.info(f"  - 目标任务数: {target_count}")
    logger.info(f"  - 实际工作进程数: {actual_workers}")
    logger.info(f"  - 每进程CPU核心数: {actual_cores_per_worker}")
    logger.info(f"  - 总使用核心数: {actual_workers * actual_cores_per_worker}")
    
    return actual_workers, actual_cores_per_worker 