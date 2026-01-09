#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并SMILES片段文件的脚本

本脚本用于将多个SMILES片段文件(.smi)按预定顺序
合并到一个单一的输出文件中。 ZINC
"""

import os

def combine_smi_files():
    """
    按指定顺序读取多个.smi文件,并将它们的内容合并写入一个新的.smi文件。
    """
    
    #datasets/source_compounds/Fragment_MW_100_to_150.smi
    #datasets/source_compounds/Fragment_MW_up_to_100.smi
    input_filenames = [
        "../source_compounds/Fragment_MW_up_to_100.smi",
        "../source_compounds/Fragment_MW_100_to_150.smi",
        "../source_compounds/Fragment_MW_150_to_200.smi",
        "../source_compounds/Fragment_MW_200_to_250.smi"
    ]
   
    output_filename = "ZINC250k.smi"

    print(f"开始合并文件到 {output_filename}...")

    try:
       
        with open(output_filename, 'w', encoding='utf-8') as outfile:           
            for filename in input_filenames:                
                if not os.path.exists(filename):
                    print(f"--> 警告: 文件 '{filename}' 不存在，已跳过。")
                    continue

                print(f"--> 正在添加文件: {filename}")              
                with open(filename, 'r', encoding='utf-8') as infile:                    
                    content = infile.read()                   
                    outfile.write(content)                    
                   
                    if content and not content.endswith('\n'):
                        outfile.write('\n')

        print(f"\n合并成功!所有内容已保存到 {output_filename}。")

    except IOError as e:
        print(f"发生文件读写错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    combine_smi_files()