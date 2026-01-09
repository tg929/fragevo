#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragEvo.
"""
import argparse
import json
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

from operations.operations_execute_fragevo import FragEvoWorkflowExecutor
from utils.cpu_utils import calculate_optimal_workers, get_available_cpu_cores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FRAGEVO_MAIN")

def run_workflow_for_receptor(
    config_path: str,
    receptor_name: Optional[str],
    output_dir: Optional[str],
    num_processors: int,
) -> Tuple[str, bool]:
    """
    Run the full FragEvo workflow for a single receptor (process-safe wrapper).

    Args:
        config_path: Path to the JSON config file.
        receptor_name: Receptor name.
        output_dir: Output root directory.
        num_processors: CPU cores allocated to this worker.

    Returns:
        Tuple[str, bool]: (receptor_display_name, success)
    """
    receptor_display_name = receptor_name or "default"
    logger.info("=" * 80)
    logger.info(
        "Starting worker process for receptor '%s' (PID: %s)",
        receptor_display_name,
        os.getpid(),
    )
    logger.info("Allocated CPU cores: %s", num_processors)
    logger.info("=" * 80)
    try:
        executor = FragEvoWorkflowExecutor(
            config_path=config_path,
            receptor_name=receptor_name,
            output_dir_override=output_dir,
            num_processors_override=num_processors,
        )
        success = executor.run_complete_workflow()
        if success:
            logger.info(
                "Worker finished successfully: receptor '%s' (PID: %s)",
                receptor_display_name,
                os.getpid(),
            )
            logger.info("Lineage tracking saved to: %s", executor.lineage_tracker_path)
            logger.info(
                "EvoMol exports: %s/pop.csv, %s/removed_ind_act_history.csv",
                executor.output_dir,
                executor.output_dir,
            )
        else:
            logger.error(
                "Worker failed: receptor '%s' (PID: %s)",
                receptor_display_name,
                os.getpid(),
            )
        return receptor_display_name, success

    except Exception as e:
        logger.critical(
            "Unhandled exception while running workflow for receptor '%s': %s",
            receptor_display_name,
            e,
            exc_info=True,
        )
        return receptor_display_name, False


def main() -> None:
    """
    CLI entry point for running the FragEvo workflow.

    Parallelism is fully controlled by the config file; CLI flags only select
    receptors and optionally override the output directory.
    """
    parser = argparse.ArgumentParser(
        description="FragEvo hybrid molecule generation entry point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='fragevo/config_fragevo.json',
        help='Path to the main config JSON file',
    )
    parser.add_argument(
        '--receptor',
        type=str,
        default=None,
        help='(Optional) Receptor name to run (None uses the default receptor)',
    )
    parser.add_argument(
        '--all_receptors',
        action='store_true',
        help='(Optional) Run all receptors listed under receptors.target_list in the config',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='(Optional) Output root directory override',
    )

    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.critical("Config file not found: %s", args.config)
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        logger.critical("Failed to parse config JSON: %s (%s)", args.config, exc)
        raise SystemExit(1)

    if args.all_receptors:
        logger.info("Detected --all_receptors; running all receptors from the config")
        receptor_map = config.get('receptors', {}).get('target_list', {})
        receptors_to_run = list(receptor_map.keys())
    else:
        receptors_to_run = [args.receptor]

    if not receptors_to_run:
        logger.critical("No receptors selected to run.")
        raise SystemExit(1)

    receptors_to_log = [name or "default" for name in receptors_to_run]
    logger.info("Receptors to run: %s", receptors_to_log)

    performance_config = config.get('performance', {})
    parallel_enabled = bool(performance_config.get('parallel_processing', False))
    max_workers_config = performance_config.get('max_workers', -1)
    inner_processors_config = performance_config.get('number_of_processors', -1)
    if max_workers_config is None:
        max_workers_config = -1
    if inner_processors_config is None:
        inner_processors_config = -1
    
    num_receptors = len(receptors_to_run)
    
    logger.info("Parallel settings (from config):")
    logger.info("  - parallel_processing: %s", "enabled" if parallel_enabled else "disabled")
    logger.info("  - max_workers: %s", max_workers_config)
    logger.info("  - number_of_processors: %s", inner_processors_config)

    successful_runs: List[str] = []
    failed_runs: List[str] = []

    if not parallel_enabled or num_receptors <= 1:
        logger.info("=" * 60)
        logger.info("Running in serial mode")
        logger.info("=" * 60)
        
        if inner_processors_config == -1:
            available_cores, _cpu_usage = get_available_cpu_cores()
            cores_per_receptor = available_cores
        else:
            cores_per_receptor = inner_processors_config
            
        logger.info("CPU cores per receptor: %s", cores_per_receptor)
        
        for receptor_name in receptors_to_run:
            receptor_display_name, success = run_workflow_for_receptor(
                args.config, receptor_name, args.output_dir, cores_per_receptor
            )
            (successful_runs if success else failed_runs).append(receptor_display_name)
    else:
        logger.info("=" * 60)
        logger.info("Running in parallel mode")
        logger.info("Detecting available CPU resources...")
        logger.info("=" * 60)
        
        available_cores, cpu_usage = get_available_cpu_cores()        

        if max_workers_config == -1 and inner_processors_config == -1:
            max_workers, cores_per_worker = calculate_optimal_workers(
                target_count=num_receptors,
                available_cores=available_cores,
                cores_per_worker=-1
            )
        elif max_workers_config == -1:
            max_possible_workers = max(1, available_cores // inner_processors_config)
            max_workers = min(num_receptors, max_possible_workers)
            cores_per_worker = inner_processors_config
        elif inner_processors_config == -1:
            max_workers = min(max_workers_config, num_receptors)
            cores_per_worker = max(1, available_cores // max_workers)
        else:
            max_workers = min(max_workers_config, num_receptors)
            cores_per_worker = inner_processors_config
        
        logger.info("Parallel execution plan:")
        logger.info("  - concurrent receptors: %s", max_workers)
        logger.info("  - CPU cores per receptor: %s", cores_per_worker)
        logger.info("  - estimated total cores: %s", max_workers * cores_per_worker)
        logger.info("  - current CPU usage: %.1f%%", cpu_usage)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_workflow_for_receptor,
                    args.config,
                    receptor_name,
                    args.output_dir,
                    cores_per_worker
                ): receptor_name for receptor_name in receptors_to_run
            }
            
            for future in as_completed(futures):
                receptor_display_name, success = future.result()
                (successful_runs if success else failed_runs).append(receptor_display_name)

    logger.info("=" * 80)
    logger.info("All FragEvo workflows finished")
    logger.info("Successful receptors (%s): %s", len(successful_runs), successful_runs)
    logger.info("Failed receptors (%s): %s", len(failed_runs), failed_runs)
    logger.info("=" * 80)

    raise SystemExit(1 if failed_runs else 0)

if __name__ == "__main__":
    # On Windows or macOS, using 'spawn' (or 'forkserver') avoids issues with forking.
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
