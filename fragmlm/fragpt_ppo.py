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
        Agent = GPT(mconf)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.load_state_dict(torch.load(os.path.join(path_here, weight_path)))
            Agent.load_state_dict(torch.load(restore_agent_from))
            Prior.cuda()
            Agent.cuda()

        else:
            Prior.load_state_dict(torch.load(os.path.join(path_here, weight_path), map_location=lambda storage, loc: storage))
            Agent.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        Prior.eval()
        Prior.requires_grad_(False)
        print((f'Agent总参数量：{sum(p.numel() for p in Agent.parameters() if p.requires_grad) / 1e6:.3f} 百万'))

        optimizer = torch.optim.AdamW(Agent.parameters(), lr=config['learning_rate'])
        scaler = GradScaler()
        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(tokenizer)

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
                    ref_probs = ref_probs * mask[:, 1:]
                logits, _, _ = Agent(seqs, tokenizer)
                probs = logits_to_probs(logits, seqs)
                probs = probs * mask[:, 1:]

            ref_sentence_probs = ref_probs.sum(dim=1)
            agent_sentence_probs = probs.sum(dim=1)

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

            # Calculate augmented likelihood
            augmented_likelihood = ref_sentence_probs.float() + config['sigma'] * Variable(score).float()
            loss = torch.pow((augmented_likelihood - agent_sentence_probs), 2)

            # Experience Replay
            # First sample
            if config['experience_replay'] and len(experience) > config['experience_replay']:
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay'])
                exp_mask = (exp_seqs != tokenizer.pad_token_id)
                with torch.amp.autocast('cuda'):
                    exp_logits, _, _ = Agent(exp_seqs, tokenizer)
                    exp_probs = logits_to_probs(exp_logits, exp_seqs)
                    exp_probs = exp_probs * exp_mask[:, 1:]
                    exp_agent_sentence_probs = exp_probs.sum(dim=1)
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_sentence_probs), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_sentence_probs = torch.cat((agent_sentence_probs, exp_agent_sentence_probs), 0)

            # Then add new experience
            prior_likelihood = ref_sentence_probs.data.cpu().numpy()
            new_experience = zip(sentence, score, prior_likelihood)
            experience.add_experience(new_experience)

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


