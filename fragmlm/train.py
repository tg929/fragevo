import torch
from utils.train_utils import seed_all
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
# from dataset import SmileDataset, SmileCollator
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
from trainer import TrainerConfig, Trainer
import datasets


def ddp_setup(rank: int, world_size: int):
    #初始化使用nccl后端
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12337"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6, 7, 8, 9"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args):
    #设置随机种子的值
    seed_value = 42
    seed_all(seed_value)
    ddp_setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')  # 逻辑编号 cuda:0 对应 os.environ["CUDA_VISIBLE_DEVICES"]中的第一个gpu
    batch_size = args.batch_size

    train_names = "train"
    val_names = "validation"
    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    raw_datasets = datasets.load_from_disk(args.dataset_path)
    traindata = SmileDataset(raw_datasets, data_type=train_names, tokenizer=tokenizer)
    validdata = SmileDataset(raw_datasets, data_type=val_names, tokenizer=tokenizer)

    collator = SmileCollator(tokenizer)
    train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(traindata), collate_fn=collator, num_workers=5)
    valid_dataloader = DataLoader(validdata, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(validdata), collate_fn=collator, num_workers=5)

    """ 
    GPT-1 like network roughly 125M params
    n_layer = 12
    n_head = 12
    n_embd = 768
    """
    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    tconf = TrainerConfig(max_epochs=1, batch_size=batch_size, learning_rate=5e-4, lr_decay=True, warmup_iters=273373,
                          final_iters=2733726, ckpt_path=f'./weights/fragpt', generate=False)
    trainer = Trainer(model, train_dataloader, valid_dataloader, tconf, tokenizer, device, rank)
    # wandb.init(project="lig_gpt", name=args.run_name)
    # wandb.init(mode="disabled")
    trainer.train()

    destroy_process_group()


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dataset_path', type=str, help="path to dataset file.")

    opt = parser.parse_args()
    # wandb.init(mode="disabled")
    # wandb.init(project="lig_gpt", name=opt.run_name)
    world_size = opt.world_size
    mp.spawn(main, args=(world_size, opt), nprocs=world_size)
