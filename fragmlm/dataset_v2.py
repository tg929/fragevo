import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SmileDataset(Dataset):

    def __init__(self, args, data_type, tokenizer):
        self.src_data_path = args.dataset_path + '/' + f"src-{data_type}_linker_without_property"
        self.tgt_data_path = args.dataset_path + '/' + f"linker-{data_type}_linker_without_property"

        with open(self.src_data_path, "r", encoding="utf-8") as f:
            self.src_data = [line.strip() for line in f.readlines()]

        with open(self.tgt_data_path, "r", encoding="utf-8") as f:
            self.tgt_data = [line.strip() for line in f.readlines()]

        self.tokenizer = tokenizer

    def __len__(self):
        assert len(self.src_data) == len(self.tgt_data)
        return len(self.src_data)

    def __getitem__(self, idx):
        src_smiles, tgt_smiles = self.tokenizer.bos_token + self.src_data[idx], self.tgt_data[idx] + self.tokenizer.eos_token
        # concat_data = src_smiles + '[SEP]' + tgt_smiles   # ?为什么极度增加显存使用
        concat_data = src_smiles + '[ANS]' + tgt_smiles

        smiles_data = self.tokenizer.encode(concat_data, add_special_tokens=False)
        smiles_data = torch.tensor(smiles_data, dtype=torch.long)

        return smiles_data


class SmileCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        smiles_datas = batch

        # Padding (batch, max_len)
        padded_smiles_datas = pad_sequence(smiles_datas, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Generate padding mask
        # Mask is True for non-padding tokens and False for padding tokens
        # padding_mask = (padded_smiles_datas != self.tokenizer.pad_token_id).float()

        x = padded_smiles_datas[:, :-1]
        y = padded_smiles_datas[:, 1:]
        # padding_mask = padding_mask[:, 1:]
        return x, y
