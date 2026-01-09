"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.train_utils import Variable

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn. MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

        self.k_cache, self.v_cache = None, None

    def forward(self, x, kv_cache=False, current_idx=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        q, k = rotary_position_embedding(q, k, current_idx)

        if kv_cache and self.eval():
            if current_idx is None:
                self.k_cache, self.v_cache = None, None
            if T == 1 and all(cache is not None for cache in (self.k_cache, self.v_cache)):  # 保证kv_cache输入的最后一个字符 T==1
                k = torch.cat((self.k_cache, k), dim=2)
                v = torch.cat((self.v_cache, v), dim=2)
            self.k_cache, self.v_cache = k, v


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)  # 下三角矩阵
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, kv_cache=False, current_idx=None):
        y, attn = self.attn(self.ln1(x), kv_cache, current_idx)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


class GPT(nn.Module):
    """  the full GPT language model """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)   # no need to weight decay
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, tokenizer, targets=None, kv_cache=False, current_idx=None):
        b, t = idx.size()

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)  # [batch_size , token_length, embedding]
        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x, kv_cache, current_idx)  # transformer * 8
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_token_id)

        return logits, loss, attn_maps  # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)

    @torch.inference_mode()
    def generate(self, idx, tokenizer, max_new_tokens, temperature=0.7, top_k=16, stream=True, rp=1., kv_cache=True, is_simulation=False):
        # rp: repetition_penalty
        index = idx.shape[1]
        init_inference = True
        while idx.shape[1] < max_new_tokens - 1:
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, tokenizer, kv_cache=kv_cache), False
            else:
                inference_res = self(idx[:, -1:], tokenizer, kv_cache=kv_cache, current_idx=idx.shape[1] - 1)

            logits, _, _ = inference_res
            logits = logits[:, -1, :]

            for token in set(idx.tolist()[0]):
                logits[:, token] /= rp

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            idx = torch.cat((idx, idx_next), dim=1)

            if is_simulation:
                if idx_next == tokenizer.eos_token_id:
                    break
            else:
                if idx_next == tokenizer.sep_token_id or idx_next == tokenizer.eos_token_id:
                    break
            if stream:
                yield idx[:, index:]

        if not stream:
            yield idx[:, index:]

    def sample(self, batch_size, tokenizer, max_new_tokens=512):
        # rp: repetition_penalty
        x = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool)
        if torch.cuda.is_available():
            x = x.cuda()
            finished = finished.cuda()

        # 生成过程
        for step in range(max_new_tokens):
            logits, _, _ = self(x, tokenizer)
            logits = logits[:, -1, :]  # 取最后一个 token 的 logits，形状 (batch_size, vocab_size)
            prob = torch.nn.functional.softmax(logits, dim=1)
            x_nest = torch.multinomial(prob, num_samples=1)  # 采样得到下一个 token
            x = torch.cat((x, x_nest), dim=1)  # 将采样到的 token 拼接到序列中

            # 检查是否生成 EOS
            eos_mask = (x_nest.squeeze(1) == tokenizer.eos_token_id)
            finished |= eos_mask
            if finished.all():
                break

        # 向量化构造 mask
        # eos_mask_full：记录每个位置是否为 EOS token
        batch, seq_len = x.shape
        eos_mask_full = (x == tokenizer.eos_token_id)
        # 判断每行是否有 EOS
        has_eos = eos_mask_full.any(dim=1)
        # 对于有 EOS 的行，使用 argmax 得到第一次出现的 EOS 的索引，
        # 对于没有 EOS 的行，令第一次 EOS 索引为最后一个位置
        first_eos_idx = torch.where(
            has_eos,
            torch.argmax(eos_mask_full.int(), dim=1),
            torch.full((batch,), seq_len - 1, dtype=torch.long, device=x.device)
        )
        # 构造一个行向量 [0, 1, ..., seq_len-1]，并与 first_eos_idx 比较
        idxs = torch.arange(seq_len, device=x.device).unsqueeze(0)  # 形状 (1, seq_len)
        mask = idxs <= first_eos_idx.unsqueeze(1)  # 每行：索引小于等于第一次 EOS 的位置为 True

        return x, mask


def rotary_position_embedding(q, k, current_idx=None):
    """
    Rotary Position Embedding (RoPE) for queries and keys.

    Args:
        q: tensor for queries of shape (batch_size, num_heads, seq_len, dim)
        k: tensor for keys of shape (batch_size, num_heads, seq_len, dim)

    Returns:
        Rotated queries and keys
    """
    batch_size, num_heads, seq_len, dim = q.size()

    # Begin of sinusoidal_position_embedding content
    # 序列对应的位置序号
    if current_idx:
        position = torch.tensor([current_idx], dtype=torch.float).unsqueeze(-1).to(q.device)
    else:
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)

    # q维度上的theta值
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)).to(q.device)

    pos_emb = position * div_term
    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    pos_emb = pos_emb.unsqueeze(0).unsqueeze(1)
    pos_emb = pos_emb.expand(batch_size, num_heads, -1, -1)
    # End of sinusoidal_position_embedding content

    # Extract and duplicate cosine and sine embeddings
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # Create alternate versions of q and k
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    k_alternate = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.size())

    # Rotate queries and keys
    q_rotated = q * cos_emb + q_alternate * sin_emb
    k_rotated = k * cos_emb + k_alternate * sin_emb

    return q_rotated, k_rotated


