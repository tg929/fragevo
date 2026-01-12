#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragEvo workflow executor (Comp Score selection variant)
========================================================
This module reuses the full FragEvo workflow and only replaces the selection
stage with a new Comp Score based selector that computes:

  y = DS_hat * QED * SA_hat

All other steps (decomposition/masking, GPT generation, GA ops, docking,
evaluation, cleanup, etc.) remain identical to the finetune executor.
"""

import os
from typing import Optional

from .operations_execute_fragevo_finetune import FragEvoWorkflowExecutor


class FragEvoCompScoreWorkflowExecutor(FragEvoWorkflowExecutor):
    """Override only the selection stage to use Comp Score based selection."""

    def run_selection(self, parent_docked_file: str, offspring_docked_file: str, generation: int) -> Optional[str]:
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"第 {generation} 代: 使用Comp Score评分函数进行选择...")
        gen_dir = self.output_dir / f"generation_{generation}"
        next_parents_file = self.output_dir / f"generation_{generation+1}" / "initial_population_docked.smi"
        next_parents_file.parent.mkdir(exist_ok=True)

        # n_select can be set via config; script also reads config_file
        selection_config = self.config.get('selection', {})
        comp_score_settings = selection_config.get('comp_score_settings', {})
        n_select = comp_score_settings.get('n_select', None)

        selection_args = [
            '--docked_file', str(offspring_docked_file),
            '--parent_file', str(parent_docked_file),
            '--output_file', str(next_parents_file),
            '--config_file', self.config_path
        ]
        if n_select is not None:
            selection_args.extend(['--n_select', str(n_select)])

        # Call the Comp Score selector
        succeeded = self._run_script('operations/selecting/selecting_comp_score.py', selection_args)
        if not succeeded or self._count_molecules(str(next_parents_file)) == 0:
            logger.error(f"第 {generation} 代: Comp Score选择失败或未选出任何分子。")
            return None

        selected_count = self._count_molecules(str(next_parents_file))
        logger.info(f"Comp Score选择完成: 选出 {selected_count} 个分子作为下一代父代。")
        return str(next_parents_file)
