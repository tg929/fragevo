import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import re


class SmileDataset(Dataset):

    def __init__(self, raw_datasets, data_type, tokenizer):
        self.data = raw_datasets[data_type]
        self.tokenizer = tokenizer

        # self.tgt_data = self.tgt_data[0:200]

    def __len__(self):
        return len(self.data)
        # return 1000

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_input = sample['input']  # 假设 'input' 存储了 SMILES 和属性信息
        tgt_smiles = sample['mc_labels']  # 假设 'mc_labels' 是目标 SMILES

        src_smiles = self.tokenizer.bos_token + src_input + self.tokenizer.eos_token

        smiles_data = self.tokenizer.encode(src_smiles, add_special_tokens=False)
        smiles_data = torch.tensor(smiles_data, dtype=torch.long)

        return smiles_data


class SmileCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        smiles_datas = batch

        # Padding (batch, max_len)
        padded_smiles_datas = pad_sequence(smiles_datas, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        x = padded_smiles_datas[:, :-1]
        y = padded_smiles_datas[:, 1:]
        # padding_mask = padding_mask[:, 1:]

        return x, y
