执行各类主文件脚本的命令汇总：
1. GA_main.py
    测试：
    python GA_main.py --receptor parp1

2. FragEvo_rag.py
    测试：
    python FragEvo_rag.py --receptor parp1

3. FragEvo_main.py
    测试
    python FragEvo_main.py --receptor parp1 --output_dir output

4. FragEvo_main_finetune.py
    测试
    python FragEvo_main_finetune.py --receptor parp1 --output_dir output
     单目标：config_fragevo.json：修改selection.selection_mode 取值为 "single_objective"
     多目标：config_fragevo.json：修改selection.selection_mode 取值为 "multi_objective";
            并确认selection.multi_objective_settings.enhanced_strategy 为 "standard"
            
            standrad代表标准的NSGA-II多目标选择方式
