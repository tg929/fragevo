#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragEvo_compscore.py
====================
Entry point for running the FragEvo workflow using the Comp Score based
selection strategy. All other stages and scripts are identical to the
finetune pipeline; only the selection step is swapped.
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from operations.operations_execute_fragevo_compscore import FragEvoCompScoreWorkflowExecutor
from utils.cpu_utils import calculate_optimal_workers, get_available_cpu_cores


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FRAGEVO_COMPSCORE_MAIN")


def _format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    return str(timedelta(seconds=int(round(seconds))))


def run_workflow_for_receptor(
    config_path: str,
    receptor_name: Optional[str],
    output_dir: Optional[str],
    num_processors: int,
) -> Tuple[str, bool]:
    receptor_display_name = receptor_name or "default"
    try:
        executor = FragEvoCompScoreWorkflowExecutor(
            config_path=config_path,
            receptor_name=receptor_name,
            output_dir_override=output_dir,
            num_processors_override=num_processors,
        )
        return receptor_display_name, executor.run_complete_workflow()
    except Exception:
        logger.exception("Unhandled exception while running workflow for receptor '%s'", receptor_display_name)
        return receptor_display_name, False


def main() -> None:
    parser = argparse.ArgumentParser(description='FragEvo workflow with Comp Score selection')
    parser.add_argument('--config', type=str, default='fragevo/config_fragevo_compscore.json')
    parser.add_argument('--receptor', type=str, default=None)
    parser.add_argument('--all_receptors', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
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
        receptor_map = config.get('receptors', {}).get('target_list', {})
        receptors_to_run = list(receptor_map.keys())
    else:
        receptors_to_run = [args.receptor]

    perf = config.get('performance', {})
    parallel_enabled = bool(perf.get('parallel_processing', False))
    max_workers_config = perf.get('max_workers', -1)
    inner_processors_config = perf.get('number_of_processors', -1)
    if max_workers_config is None:
        max_workers_config = -1
    if inner_processors_config is None:
        inner_processors_config = -1

    if not parallel_enabled or len(receptors_to_run) <= 1:
        if inner_processors_config == -1:
            available_cores, _ = get_available_cpu_cores()
            cores_per = available_cores
        else:
            cores_per = inner_processors_config
        results = [run_workflow_for_receptor(args.config, r, args.output_dir, cores_per) for r in receptors_to_run]
    else:
        available_cores, _ = get_available_cpu_cores()
        if max_workers_config == -1 and inner_processors_config == -1:
            max_workers, cores_per = calculate_optimal_workers(
                target_count=len(receptors_to_run), available_cores=available_cores, cores_per_worker=-1
            )
        elif max_workers_config == -1:
            max_possible_workers = max(1, available_cores // inner_processors_config)
            max_workers = min(len(receptors_to_run), max_possible_workers)
            cores_per = inner_processors_config
        elif inner_processors_config == -1:
            max_workers = min(max_workers_config, len(receptors_to_run))
            cores_per = max(1, available_cores // max_workers)
        else:
            max_workers = min(max_workers_config, len(receptors_to_run))
            cores_per = inner_processors_config

        results: List[Tuple[str, bool]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(run_workflow_for_receptor, args.config, r, args.output_dir, cores_per)
                for r in receptors_to_run
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    success = [r for r, ok in results if ok]
    failed = [r for r, ok in results if not ok]
    logger.info("Comp Score selection workflow finished. success=%s failed=%s", success, failed)
    raise SystemExit(0 if not failed else 1)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    _t0 = time.perf_counter()
    try:
        main()
    finally:
        logger.info("Total elapsed time: %s", _format_elapsed(time.perf_counter() - _t0))
