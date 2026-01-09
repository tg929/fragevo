import torch
from utils.train_utils import seed_all
import os
import argparse
import json
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
import datasets
from rdkit import Chem
from utils.train_utils import get_mol
from utils.chem_utils import reconstruct
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAGMENT_GPT_DIR = os.path.dirname(os.path.abspath(__file__))



def Test(model, tokenizer, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation, device,
         output_file_path, seed,input_file=None):
    complete_answer_list = []
    valid_answer_list = []
    model.eval()
    if input_file:
        with open(input_file, 'r') as f:
            prefixes = [line.strip() for line in f if line.strip()]
    else:
        prefixes = [None]

    for input_prefix in tqdm(prefixes, desc='Processing molecules'):
        if input_prefix:
            prefix_tokens = tokenizer.encode(input_prefix, add_special_tokens=False)
            x = torch.tensor([prefix_tokens], dtype=torch.int64).to(device)
        else:
            x = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.int64).to(device)

        with torch.no_grad():
            res_y = model.generate(x, tokenizer, max_new_tokens=max_seq_len,
                                temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache,
                                is_simulation=is_simulation)
        try:
            y = next(res_y)
        except StopIteration:
            print("No answer")
            continue

        history_idx = 0
        complete_answer = f"{tokenizer.decode(x[0])}"

        while y != None:
            answer = tokenizer.decode(y[0].tolist())
            if answer and answer[-1] == '':
                try:
                    y = next(res_y)
                except:
                    break
                continue

            if not len(answer):
                try:
                    y = next(res_y)
                except:
                    break
                continue

            complete_answer += answer[history_idx:]

            try:
                y = next(res_y)
            except:
                break
            history_idx = len(answer)
            if not stream:
                break

        complete_answer = complete_answer.replace(" ", "").replace("[BOS]", "").replace("[EOS]", "")
        frag_list = complete_answer.replace(" ", "").split('[SEP]')
        try:
            frag_mol = [Chem.MolFromSmiles(s) for s in frag_list]
            mol = reconstruct(frag_mol)[0]
            if mol:
                generate_smiles = Chem.MolToSmiles(mol)
                valid_answer_list.append(generate_smiles)
                answer = frag_list
            else:
                answer = frag_list
        except:
            answer = frag_list
        complete_answer_list.append(answer)

    print(
        f"valid ratio:{len(valid_answer_list)}/{len(complete_answer_list)}={len(valid_answer_list) / len(complete_answer_list)}")
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, "w") as w:
        for j in valid_answer_list:
            w.write(j)
            w.write("\n")
    w.close()


def load_config(config_path):
    """从配置文件加载GPT相关参数"""
    if not config_path:
        return {}
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        gpt_config = config.get('gpt', {})
        if gpt_config:
            print(f"成功从配置文件加载GPT参数: {config_path}")
        else:
            print(f"配置文件中未找到GPT配置块,将使用默认参数: {config_path}")
        return gpt_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}, 将使用默认参数")
        return {}


def main_test(args):
    gpt_config = load_config(args.config_file)
    
    if args.seed:
        seed_value = int(args.seed)
    else:
        seed_value = gpt_config.get('seed', 42)
    seed_all(seed_value)
    
    device_id_str = args.device if args.device else gpt_config.get('device', '0')
    
    if torch.cuda.is_available():
        try:
            requested_id = int(device_id_str)
            if requested_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{requested_id}")
                print(f"使用GPU: {device}")
            else:
                print(f"警告: 请求的GPU ID {requested_id} 不可用 (共 {torch.cuda.device_count()} 个GPU)。将回退到 cuda:0。")
                device = torch.device("cuda:0")
        except (ValueError, TypeError):
            print(f"警告: 无效的设备ID '{device_id_str}'。将使用CPU。")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("警告: 未找到CUDA设备, 将使用CPU进行计算。")

    tokenizer = SmilesTokenizer(os.path.join(FRAGMENT_GPT_DIR, 'vocabs/vocab.txt'))
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    collator = SmileCollator(tokenizer)

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf)
    
    checkpoint_path = os.path.join(FRAGMENT_GPT_DIR, 'weights/dpo_0_400.pt')
    print(f"正在从 {checkpoint_path} 加载模型权重到设备 {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)   
    
    model.load_state_dict(checkpoint)
    model.to(device)
    start_time = time.time()
    
    if args.output_file:
        output_file_path = args.output_file
        print(f"使用指定的输出文件路径: {output_file_path}")
    else:
        output_dir = os.path.join(PROJECT_ROOT, "fragmlm/output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, f'crossovered0_frags_new_{seed_value}.smi')
        print(f"使用默认输出路径: {output_file_path}")
    
    model_settings = gpt_config.get('model_settings', {})
    max_seq_len = model_settings.get('max_seq_len', 1024)
    temperature = gpt_config.get('temperature', 1.0)
    top_k = model_settings.get('top_k', None)
    stream = model_settings.get('stream', False)
    rp = model_settings.get('rp', 1.0)
    kv_cache = model_settings.get('kv_cache', True)
    is_simulation = model_settings.get('is_simulation', True)
    
    print(f"GPT生成参数: max_seq_len={max_seq_len}, temperature={temperature}, seed={seed_value}")
    
    Test(model, tokenizer, max_seq_len=max_seq_len, temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache,
         is_simulation=is_simulation, device=device, output_file_path=output_file_path, seed=seed_value, input_file=args.input_file)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"运行时间: {elapsed_time:.4f} 秒")


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='Fragment GPT 分子生成脚本')
    parser.add_argument('--device', help='GPU设备ID (如不指定则从配置文件读取)')
    parser.add_argument('--seed', help='随机种子 (如不指定则从配置文件读取)')
    parser.add_argument('--input_file', required=True, help='批量条件文件路径（掩码片段文件）') 
    parser.add_argument('--output_file', help='输出生成分子的文件路径 (如不指定则使用默认路径)')
    parser.add_argument('--config_file', help='配置文件路径 (包含GPT相关参数)')

    opt = parser.parse_args()

    main_test(opt)