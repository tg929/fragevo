6-15 operations中脚本测试

1.对接
docking_demo_finetune.py
命令：python operations/docking/docking_demo_finetune.py --smiles_file datasets/source_compounds/naphthalene_smiles.smi --output_file output_testing_docking.smi --receptor tutorial/PARP/4r6eA_PARP1_prepared.pdbqt

2.交叉
crossover_demo_finetune.py
命令:python operations/crossover/crossover_demo_finetune.py --smiles_file datasets/source_compounds/naphthalene_smiles.smi --output_file output_testing_crossover.smi --crossover_attempts 40

过滤
filter_demo.py
python operations/filter/filter_demo.py --smiles_file output_testing_crossover.smi --output_file output_testing_filter1.smi

3.突变
mutation_demo_finetune.py
命令：python operations/mutation/mutation_demo_finetune.py --smiles_file datasets/source_compounds/naphthalene_smiles.smi --output_file output_testing_mutation.smi --num_mutations 40

mutation_test.py
python operations/mutation/mutation_test.py --input_file datasets/source_compounds/naphthalene_smiles.smi --output_file output_testing_mutation.smi 

过滤
filter_demo.py
python operations/filter/filter_demo.py --smiles_file output_testing_mutation.smi --output_file output_testing_filter2.smi

子代：种群---output_testing_filter1.smi + output_testing_filter2.smi

4.再次对接评分
docking_demo_finetune.py
命令：python operations/docking/docking_demo_finetune.py --smiles_file 子代：种群 --output_file output_testing_docking1.smi --receptor tutorial/PARP/4r6eA_PARP1_prepared.pdbqt

5.选择
filter_demo.py 
命令：python operations/selecting/molecular_selection.py --docked_file output_testing_docking.smi+output_testing_docking1.smi --n_select 115

6.交叉
7.突变
8.过滤
9.新子代 评分等
10.选择

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

6-16 更新参数配置文件之后执行各个脚本

1.

2.交叉
python operations/crossover/crossover_demo_finetune.py --smiles_file datasets/source_compounds/naphthalene_smiles.smi --output_file output_testing_crossover.smi 
默认参数：和autogrow一致  50个新分子

3.

4.突变
python operations/mutation/mutation_demo_finetune.py --smiles_file datasets/source_compounds/naphthalene_smiles.smi --output_file output_testing_mutation.smi 
默认参数：all_rxns；50个新生成分子 ；其他一致于autogrow

5.选择
python operations/selecting/molecular_selection.py --docked_file test_selection.smi --parent_file test_parent.smi --output_file test_config_output.smi
默认参数：同autogrow




