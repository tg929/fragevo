import torch
from utils.train_utils import seed_all
import os
import argparse
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
import datasets
from rdkit import Chem
from fragment_utils import reconstruct
from mcts import MCTSConfig, MolecularProblemState, MCTS
from utils.chem_utils import sentence2mol
from rdkit import rdBase
from utils.docking.docking_utils import DockingVina
from tqdm import tqdm
import pandas as pd


# 禁用所有日志信息
rdBase.DisableLog('rdApp.warning')


def Test(model, tokenizer, device, output_file_path):
    target_answer_list = []
    mcts_answer_list = []
    model.eval()
    predictor = DockingVina('parp1')
    # 找到第一个分隔符
    # indices = torch.nonzero(x.squeeze(0) == 13, as_tuple=True)[0]
    # first_index = indices[0].item()
    # x = x[:, :first_index + 1]   # 取第一个片段作为输入
    x = torch.tensor([1], dtype=torch.int64).unsqueeze(0)
    x = x.to(device)
    initial_state = MolecularProblemState(model, tokenizer, predictor, x)
    # 创建一个空的DataFrame来存储数据
    columns = ['smiles', 'rv', 'rq', 'rs', 'value']
    data = []
    with torch.no_grad():
        for i in tqdm(range(1000)):
            bos = x
            action, smiles_answer, has_end_token = initial_state.generate_fragment(
                cur_molecule=bos,
                max_seq_len=1024,
                temperature=0.8,
                top_k=None,
                stream=False,
                rp=1.0,
                kv_cache=True,
                is_simulation=True
            )
            _, smiles = sentence2mol(smiles_answer)
            (rv, rq, rs), value = initial_state.get_reward(smiles)
            # 将结果添加到data列表中
            data.append([smiles, rv, rq, rs, value])

    # 将data转换为DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 将DataFrame写入CSV文件
    df.to_csv('./smiles_rewards.csv', index=False)



def main_test(args):
    # 设置随机种子的值
    seed_value = 42
    seed_all(seed_value)
    rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device(f'cuda:{0}')  # 逻辑编号 cuda:0 对应 os.environ["CUDA_VISIBLE_DEVICES"]中的第一个gpu
    batch_size = 1

    test_names = "test"

    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(f'./weights/linkergpt.pt', weights_only=True)
    # checkpoint = torch.load(f'/data1/yzf/molecule_generation/a/LinkerGPT/weights/{args.run_name}.pt', weights_only=True)
    model.load_state_dict(checkpoint)
    start_time = time.time()
    Test(model, tokenizer, device=device, output_file_path="./output")
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"运行时间: {elapsed_time:.4f} 秒")


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--run_name', default='linkergpt', help='name of .pt file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()
    # wandb.init(mode="disabled")
    # wandb.init(project="lig_gpt", name=opt.run_name)
    world_size = opt.world_size
    # mp.spawn(main, args=(world_size, opt), nprocs=world_size)

    main_test(opt)

