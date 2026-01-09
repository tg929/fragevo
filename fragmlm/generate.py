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
from utils.train_utils import get_mol
from fragment_utils import reconstruct


def Test(model, test_dataloader, tokenizer, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation, device, output_file_path):
    loader = test_dataloader
    complete_answer_list = []
    valid_answer_list = []
    target_answer_list = []
    model.eval()
    for x, target in loader:
        # place data on the correct device
        indices = torch.nonzero(x.squeeze(0) == 13, as_tuple=True)[0]
        first_index = indices[0].item()
        x = x[:, :first_index+1]
        x = x.to(device)
        # pbar.set_description(f"iter {it}")
        target_answer_list.append(target[0])
        with torch.no_grad():
            print("input_sequence:", tokenizer.decode(x[0]))  # [batch, len]
            print("label_sequence:", tokenizer.decode(target[0]))
            res_y = model.generate(x, tokenizer, max_new_tokens=max_seq_len,
                                   temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache, is_simulation=is_simulation)
            print('[A]: ', end='')
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            complete_answer = f"{tokenizer.decode(x[0])}"  # 用于保存整个生成的句子

            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue
                # print(answer)
                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                # 保存生成的片段到完整回答中
                complete_answer += answer[history_idx:]

                print(answer[history_idx:], end='', flush=True)
                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            complete_answer = complete_answer.replace(" ", "").replace("[BOS]", "")
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
            print("\n")

    print(f"valid ratio:{len(valid_answer_list)}/{len(complete_answer_list)}={len(valid_answer_list) / len(complete_answer_list)}")
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)
    with open(os.path.join(output_file_path, 'complete_answer'), "w") as w:
        for j in complete_answer_list:
            if not isinstance(j, str):
                j = str(j)
            w.write(j)
            w.write("\n")
    w.close()
    with open(os.path.join(output_file_path, 'valid_answer'), "w") as w:
        for j in valid_answer_list:
            w.write(j)
            w.write("\n")
    w.close()


def main_test(args):
    #设置随机种子的值
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
    
    raw_datasets = datasets.load_from_disk(args.dataset_path)
    testdata = SmileDataset(raw_datasets, data_type=test_names, tokenizer=tokenizer)

    collator = SmileCollator(tokenizer)
    test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(f'./weights/linkergpt.pt', weights_only=True)
    # checkpoint = torch.load(f'/data1/yzf/molecule_generation/a/LinkerGPT/weights/{args.run_name}.pt', weights_only=True)
    model.load_state_dict(checkpoint)
    start_time = time.time()
    Test(model, test_dataloader, tokenizer, max_seq_len=1024, temperature=0.7, top_k=16, stream=False, rp=1., kv_cache=True, is_simulation=True, device=device, output_file_path="./output")
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
    parser.add_argument('--dataset_path', type=str, help="path to dataset file.")

    opt = parser.parse_args()
    # wandb.init(mode="disabled")
    # wandb.init(project="lig_gpt", name=opt.run_name)
    world_size = opt.world_size
    # mp.spawn(main, args=(world_size, opt), nprocs=world_size)

    main_test(opt)

