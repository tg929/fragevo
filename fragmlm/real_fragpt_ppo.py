import os
import sys
import numpy as np

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from optimizer import BaseOptimizer
from utils.train_utils import Variable
from utils.chem_utils import sentence2mol, unique
from model import GPT
from data_structs import Experience
import torch
from torch.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F


class GPTWithValueHead(GPT):
    """在原有GPT基础上增加价值函数头"""

    def __init__(self, mconf):
        super().__init__(mconf)
        self.value_head = nn.Sequential(
            nn.Linear(mconf.n_embd, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, seqs, tokenizer, targets=None, kv_cache=False, current_idx=None):
        tok_emb = self.tok_emb(seqs)  # [B, L, D]
        x = self.drop(tok_emb)

        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x, kv_cache, current_idx)  # transformer * 8
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        value = self.value_head(x)

        return logits, value, attn_maps

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


def logits_to_probs(logits, seqs):
    # logits shape: (batch_size, seq_len, vocab_size)
    # seqs shape: (batch_size, seq_len)
    x = logits[:, :-1, :]
    labels = seqs[:, 1:]
    log_probs = torch.nn.functional.log_softmax(x, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


class FRAGPT_Optimizer(BaseOptimizer):

    def __init__(self, model):
        super().__init__()
        self.model_name = "FRAGPT"
        self.model = model

    def _optimize(self, oracle, config, mconf, tokenizer, weight_path):

        self.oracle.assign_evaluator(oracle)

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from = os.path.join(path_here, weight_path)
        restore_agent_from = restore_prior_from

        Prior = GPT(mconf)
        Agent = GPTWithValueHead(mconf)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.load_state_dict(torch.load(os.path.join(path_here, weight_path)))
            Agent.load_state_dict(torch.load(restore_agent_from), strict=False)
            Prior.cuda()
            Agent.cuda()

        else:
            Prior.load_state_dict(torch.load(os.path.join(path_here, weight_path), map_location=lambda storage, loc: storage))
            Agent.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage, strict=False))

        # We dont need gradients with respect to Prior
        Prior.eval()
        Prior.requires_grad_(False)
        print((f'Agent总参数量：{sum(p.numel() for p in Agent.parameters() if p.requires_grad) / 1e6:.3f} 百万'))

        optimizer = torch.optim.AdamW(Agent.parameters(), lr=config['learning_rate'])
        scaler = GradScaler()
        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        # experience = Experience(tokenizer)

        print("Model initialized, starting training...")

        step = 0
        patience = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            with torch.amp.autocast('cuda'):
                # Sample from Agent
                with torch.no_grad():
                    seqs, mask = Agent.sample(config['batch_size'], tokenizer)
                    ref_logits, _, _ = Prior(seqs, tokenizer)
                    ref_probs = logits_to_probs(ref_logits, seqs)
                    # ref_probs = ref_probs * mask[:, 1:]

                    old_logits, old_value, _ = Agent(seqs, tokenizer)
                    old_probs = logits_to_probs(old_logits, seqs)
                    # old_probs = old_probs * mask[:, 1:]

            smiles = []
            sentence = []
            for seq in seqs:
                eos_index = (seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_index.numel() > 0:
                    seq = seq[:eos_index[0] + 1]  # 只取第一个 [EOS] 的位置
                smi = tokenizer.decode(seq)
                sentence.append(smi)
                smiles.append(sentence2mol(smi, RemoveStereo=True)[1])
            score = np.array(self.oracle(smiles))

            if self.finish:
                print('max oracle hit')
                break

            # early stopping
            if len(self.oracle) > 1000 and oracle.name != 'valsartan_smarts':
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= 5:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # compute_ppo_loss
            for _ in range(config['ppo_epochs']):
                logits, values, _ = Agent(seqs)
                cur_probs = logits_to_probs(logits, seqs)

                # 1. 策略梯度损失（Clipped Surrogate Loss）
                ratio = torch.exp(cur_probs - old_probs)
                advantages = (Variable(score) - old_value.detach()).unsqueeze(1)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * advantages
                policy_loss = -torch.min(surr1, surr2) * mask[:, 1:]
                policy_loss = policy_loss.sum() / mask[:, 1:].sum()

                # 2. 价值函数损失
                value_targets = Variable(score).unsqueeze(1).expand(-1, values.size(1))
                value_loss = F.mse_loss(values, value_targets, reduction='none') * batch['masks']
                value_loss = value_loss.sum() / batch['masks'].sum()

                # 3. KL散度约束（防止偏离Prior太远）
                kl_div = (batch['ref_log_probs'] - current_log_probs) * batch['masks']
                kl_loss = kl_div.sum() / batch['masks'].sum()

                # 4. 熵奖励（鼓励探索）
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) * batch['masks']
                entropy_bonus = -entropy.sum() / batch['masks'].sum()  # 负号因为要最大化熵

                # 合并损失项
                total_loss = (
                        policy_loss +
                        0.5 * value_loss +
                        kl_penalty * kl_loss +
                        0.01 * entropy_bonus
                )


            # # Experience Replay
            # # First sample
            # if config['experience_replay'] and len(experience) > config['experience_replay']:
            #     exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay'])
            #     exp_mask = (exp_seqs != tokenizer.pad_token_id)
            #     with torch.amp.autocast('cuda'):
            #         exp_logits, _, _ = Agent(exp_seqs, tokenizer)
            #         exp_probs = logits_to_probs(exp_logits, exp_seqs)
            #         exp_probs = exp_probs * exp_mask[:, 1:]
            #         exp_agent_sentence_probs = exp_probs.sum(dim=1)
            #     exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
            #     exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_sentence_probs), 2)
            #     loss = torch.cat((loss, exp_loss), 0)
            #     agent_sentence_probs = torch.cat((agent_sentence_probs, exp_agent_sentence_probs), 0)

            # Then add new experience
            prior_likelihood = ref_sentence_probs.data.cpu().numpy()
            # new_experience = zip(sentence, score, prior_likelihood)
            # experience.add_experience(new_experience)

            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            # loss_p = - (1 / agent_sentence_probs).mean()
            # loss += 5 * 1e3 * loss_p
            print(loss)


            # Calculate gradients and make an update to the network weights

            loss = loss / config['accumulation_steps']
            scaler.scale(loss).backward()

            if (step + 1) % config['accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(Agent.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Convert to numpy arrays so that we can print them
            augmented_likelihood = augmented_likelihood.data.cpu().numpy()
            agent_sentence_probs = agent_sentence_probs.data.cpu().numpy()
            loss = loss.item()
            step += 1


