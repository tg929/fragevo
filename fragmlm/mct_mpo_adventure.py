import re
from utils.chem_utils_ import get_morgan_fingerprint, is_too_similar_to_children, sentence2mol, get_sa, get_qed
import time
import numpy as np
from collections import deque
from rdkit import Chem
import torch
from tqdm import tqdm
import pickle
from tdc import Oracle
import json


best_score = -1e8
best_smi = None

'''
改了温度0.8->1.0
value_weight = 0
fastrollout_weight = 1.0
reward中加入max
rv放在判断rq和rs后计算
只访问可扩展结点
c_param = 5
'''


def oracle_predict(oracle_name, smi):
    predictor = Oracle(name=oracle_name)
    return predictor(smi)


def top_auc(buffer, top_n, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


def print_best():
    global best_score
    global best_smi
    print(best_score)
    print(best_smi)


class MCTSConfig:
    # optimization parameters
    value_weight = 0  # weight of value in the total reward. 0 means no value.
    search_time = 10000  # total search times (equal or larger than than the number of nodes expanded)
    min_terminals = -1  # minimum number of terminals must search
    max_split_depth = 10  # maximum depth to split the tree. If larger, only single path will be expanded. If -1, no limit. This is a piror knowledge of the problem.
    init_children = 20  # initial number of children to expand at the root node. if -1, use N_TOTAL_CHILDREN. This is a piror knowledge of the problem.
    n_total_children = 8  # number of children to expand at each node
    c_param = 5  # exploration parameter
    width_increase_factor = 2  # increase the width of the tree by this factor in Adaptive child allocation

    add_value_weight = 0.0
    n_simulations = 1
    fastrollout_weight = 1.0

    greedy_path = False
    max_n_repeat = 5
    freq_log = 100


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MolecularProblemState:

    def __init__(self,
                 model,
                 tokenizer,
                 predictor,
                 cur_molecule=None,  # 当前分子
                 cur_step=0,  # 当前步骤
                 max_steps=10,  # 最大生成步骤
                 is_terminate=False,  # 是否为终止状态
                 rewards=None,  # 奖励列表
                 has_optimized=False):  # 是否已进行优化
        """
        初始化分子问题状态，用于分子生成或优化任务。
        """
        self.predictor = predictor
        self.cur_molecule = cur_molecule
        self.model = model
        self.tokenizer = tokenizer
        sentence = self.tokenizer.decode(self.cur_molecule[0])
        self.cur_sentence = sentence
        self.cur_step = cur_step
        self.max_steps = max_steps
        self.is_terminate = is_terminate
        self.rewards = rewards if rewards is not None else []
        self.has_optimized = has_optimized

    def get_cur_molecule(self):
        return self.cur_molecule

    def get_cur_step(self):
        return self.cur_step

    def is_terminal(self):
        """
        判断是否终止：
          - 如果已经检测到SMILES (或其他判定条件) 则终止
          - 或者已经达到最大生成步数
          - 或者 is_terminate 被手动置为 True
        """
        has_eos = self.check_eos_exist()
        max_lines_reached = self.cur_step >= self.max_steps
        return has_eos or max_lines_reached or self.is_terminate

    def check_eos_exist(self):
        """
        检测当前输出中是否已经出现了 SMILES 标记或其他判定条件
        这里以简单的正则或关键字 "SMILES:" 判断为例。
        """
        # 示例：用来匹配类似 “SMILES: C1=CC=CC=C1” 这样的字符串
        if "[EOS]" in self.cur_sentence:
            return True
        else:
            return False

    @staticmethod
    def extract_smiles(completion):
        """
        从文本中提取 SMILES。
        如果未能匹配到，则返回 INVALID_ANS。
        """
        SMILES_RE = re.compile(r"(?:SMILES:\s*)([A-Za-z0-9@+\-\[\]\(\)=#$%]+)")
        match = SMILES_RE.search(completion)
        if match:
            return match.group(1).strip()
        else:
            return "<INVALID_SMILES>"

    def is_correct(self):
        """
        若存在参考 SMILES (self.answer)，可在此做简单比较/校验。
        例如：
        1. 直接字符串对比
        2. 或者使用 RDKit 等工具对分子做同一性判断（需要另行安装与配置）
        """
        predicted_smiles = self.extract_smiles(self.cur_molecule)
        if predicted_smiles == "<INVALID_SMILES>":
            return False
        # 简单示例：直接字符串比较
        return predicted_smiles

    def get_value(self):
        """
        计算分子性质得分 (示例：使用RDKit的QED作为分子打分)。
        如果SMILES非法，则返回负分以示惩罚。
        """
        _, smiles = sentence2mol(self.cur_sentence)
        value = self.get_reward(smiles)
        return value

    def get_reward(self, smiles):
        if smiles is None:
            return -1.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1.0

        reward = oracle_predict(self.predictor, smiles)
        if not isinstance(reward, float):
            a = 1
        return reward

    def cond_actions(self, to_end=False, is_greedy=False):
        """
        执行一次“只生成一步”的动作。
        可设置 is_greedy=True 做贪心解码等。
        """
        # 这里简化，不区分 simulation / real
        # 如果要区分，可以再添加参数
        n_attempts = 5
        for attempt in range(n_attempts):
            try:
                if to_end:
                    action, smiles_answer, has_end_token = self.action2end(is_greedy=is_greedy)  # 返回的是token对应idx的列表
                else:
                    action, smiles_answer, has_end_token = self.actions(is_greedy=is_greedy)
                    if len(action) == 0:
                        continue
                return action, smiles_answer, has_end_token
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f'Retry {attempt}, error: {type(e).__name__}', flush=True)
                    continue
                else:
                    raise e

    def actions(self, is_greedy=False):
        """
        只做一次推断调用，使用你自己的generate函数。
        """
        temperature = 0.0 if is_greedy else 1.0
        # 假设你自己实现了一个 my_generate 函数，可以直接调用
        action, smiles_answer, has_end_token = self.generate_fragment(
            cur_molecule=self.cur_molecule,
            max_seq_len=1024,
            temperature=temperature,
            top_k=None,
            stream=False,
            rp=1.0,
            kv_cache=True,
            is_simulation=False
        )
        return action, smiles_answer, has_end_token

    def take_action(self, action):
        """
        将生成的新文本拼接到 cur_molecule 中，更新状态。
        """
        new_answer = torch.as_tensor(action, dtype=self.cur_molecule.dtype, device=self.cur_molecule.device).unsqueeze(
            0)
        next_state = MolecularProblemState(
            model=self.model,
            tokenizer=self.tokenizer,
            predictor=self.predictor,
            cur_molecule=new_answer,
            cur_step=self.cur_step + 1,
            max_steps=self.max_steps,
            is_terminate=False  # 后面会根据 is_terminal 判定
        )
        return next_state

    def action2end(self, is_greedy):
        """
        一次性生成到结束。
        """
        temperature = 0.0 if is_greedy else 1.0
        action, smiles_answer, has_end_token = self.generate_fragment(
            cur_molecule=self.cur_molecule,
            max_seq_len=1024,
            temperature=temperature,
            top_k=None,
            stream=False,
            rp=1.0,
            kv_cache=True,
            is_simulation=True
        )

        return action, smiles_answer, has_end_token

    def take_action_end(self, is_greedy=False):
        assert is_greedy == False
        """
        一次性生成到结束版本，适用于分子GPT场景。
        """
        # 如果已经终止，就直接返回当前状态即可
        if self.is_terminal():
            return self

        # 多次重试，若真的都失败了就抛异常
        n_attempts = 20  # 可自定义
        final_action = ""
        for attempt in range(n_attempts):
            try:
                final_action, smiles_answer, has_end_token = self.action2end(is_greedy=is_greedy)
                break
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f"[take_action_end] attempt {attempt}, error: {type(e).__name__}. Retrying...")
                    continue
                else:
                    print(f"[take_action_end] All attempts failed. Error: {type(e).__name__}")
                    raise e

        # 计算生成的步骤数，如无需分行可直接视为 1 步
        # 或者如果你的 final_action 有换行，可以按换行 split
        # 这里示例用 .split('\n')
        n_steps = smiles_answer.count('[SEP]')

        # 拼接到现有答案
        answer_updated = torch.as_tensor(final_action, dtype=self.cur_molecule.dtype,
                                         device=self.cur_molecule.device).unsqueeze(0)
        # 构造一个新的 ProblemState，标记 is_terminate=True
        end_state = MolecularProblemState(
            model=self.model,
            tokenizer=self.tokenizer,
            predictor=self.predictor,
            cur_molecule=answer_updated,
            cur_step=self.cur_step + n_steps,
            max_steps=1000,  # 或者任意大值
            is_terminate=True
        )
        return end_state

    def generate_fragment(self, cur_molecule, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation):
        with torch.no_grad():
            res_y = self.model.generate(cur_molecule, self.tokenizer, max_new_tokens=max_seq_len,
                                        temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache,
                                        is_simulation=is_simulation)
            # print('[A]: ', end='')
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")

            history_idx = 0
            complete_answer = cur_molecule[0].tolist()  # 用于保存整个生成的句子

            while y != None:
                answer = y[0].tolist()
                # 保存生成的片段到完整回答中
                complete_answer += answer[history_idx:]

                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

        smiles_answer = self.tokenizer.decode(complete_answer)
        # print(smiles_answer, flush=True)
        has_end_token = False
        if "[EOS]" in smiles_answer:
            has_end_token = True

        return complete_answer, smiles_answer, has_end_token


class MonteCarloTreeSearchNode:
    """
    适配为分子GPT场景的 MCTS Node。
    如果你还有外部打分、相似度过滤等逻辑，可以保留并在里面写对 SMILES 的判断。
    """

    def __init__(self,
                 state,
                 config,
                 parent=None,
                 parent_action=None,
                 depth=0,
                 node_id=None,
                 n_repeat_by_parent=1):

        self.config = config

        # 基本节点属性
        self.state = state
        self.parent = parent
        self.parent_action = parent_action  # 该节点是由什么 action 从父节点扩展来的
        self.children = []
        self._number_of_visits = 0  # 访问次数
        self._results = []  # 回传的奖励(或评分)累加，用于计算 Q 值

        # Molecule GPT 可能需要额外的数据结构，这里仅示例
        self._values = []  # 如果有对分子序列的价值预测，可放这
        self._cached_reward = 0.  # 缓存某一次最终reward（可选）

        # 节点搜索超参
        self.depth = depth
        self.node_id = node_id
        self.n_repeat_by_parent = n_repeat_by_parent
        self.n_repeat = 0
        # 用于限制最大深度，或动态扩展子节点个数
        if self.config.max_split_depth < 0:
            # 如果是 -1，代表不限制
            self.config.max_split_depth = self.depth
        if self.depth == 0:
            self.n_total_children_adaptive = self.config.init_children if self.config.init_children > -1 else self.config.init_children
        elif self.depth > self.config.max_split_depth:
            self.n_total_children_adaptive = 1
        else:
            self.n_total_children_adaptive = self.config.n_total_children

        # 可以做一些自适应扩展或剪枝相关的变量
        self.max_q_diff = 0
        self.expandable = True

    def n(self):
        """访问次数。"""
        return self._number_of_visits

    def q(self):
        """累加的 Q 值，可简单用 sum(_results)。"""
        return np.sum(self._results)

    def result(self):
        """results列表"""
        return self._results

    def is_terminal_node(self):
        """判断状态是否已经结束（比如已经生成了完整 SMILES）。"""
        return self.state.is_terminal()

    def is_fully_expanded(self):
        """是否已经有足够的子节点，不再继续扩展。"""
        return len(self.children) >= self.n_total_children_adaptive

    def n_children(self):
        return len(self.children)

    def total_number_nodes(self):
        """计算以本节点为根的所有节点数。"""
        tot_node = 1
        for child in self.children:
            tot_node += child.total_number_nodes()
        return tot_node

    def get_ancestor_child_indices(self):
        indices = []
        current_node = self
        while current_node.parent is not None:
            index = current_node.parent.children.index(current_node)
            indices.append(index)
            current_node = current_node.parent
        return indices[::-1]

    def retrieve_origin_value(self):
        """如果有某个子节点对应的初始价值，可在此返回。"""
        return self._values[0] if len(self._values) > 0 else None

    def set_cached_reward(self, raw_value):
        self._cached_reward = raw_value

    def get_cached_reward(self):
        return self._cached_reward

    def expand(self):
        """
        选出一个可接受的新 action（即新的 SMILES 片段或下一步 Token），
        创建新的子节点并返回。
        """
        action, has_end_token, n_repeat = self.get_acceptable_action()
        self.n_repeat = n_repeat

        # 调用 ProblemState 的 take_action，将该 action 拼接到当前 SMILES/文本中，得到新状态
        next_state = self.state.take_action(action)

        # 构造新的子节点
        cur_n_children = len(self.children)
        cur_node_id = self.node_id
        child_node = MonteCarloTreeSearchNode(
            state=next_state,
            config=self.config,
            parent=self,
            parent_action=action,
            depth=self.depth + 1,
            node_id=f"{cur_node_id}-{cur_n_children}" if cur_node_id else None,
            n_repeat_by_parent=n_repeat
        )

        self.children.append(child_node)
        return child_node

    def get_acceptable_action(self):
        """
        核心函数：获取一个“合适的 action”。
        - 可以做相似度过滤
        - 可以检测 SMILES 是否已出现
        - 等等...
        """
        # 先收集本节点现有子节点的指纹
        children_fps = []
        for child in self.children:
            # 假设 child.parent_action 保存着其对应 SMILES
            child_mol, child_smiles = sentence2mol(child.state.cur_sentence)
            fp = get_morgan_fingerprint(child_mol)
            if fp is not None:
                children_fps.append(fp)
        n_repeat = 0

        # 到达最大深度，就一次性生成到结束
        to_end = self.config.max_split_depth <= (self.depth + 1)
        # 若本层还没扩展开任何子节点，且是 greedy path，可以设置 is_greedy=True
        is_greedy = self.config.greedy_path and len(self.children) == 0
        # 如果希望避免第一层 action 是空的
        # avoid_empty = self.depth == 0

        # 在这里实现一个循环重试，以保证得到一个“相似度较低”的 SMILES
        while True:
            action, smiles_answer, has_end_token = self.state.cond_actions(
                to_end=to_end,
                is_greedy=is_greedy,
            )

            new_mol, _ = sentence2mol(smiles_answer)
            # 计算 new_smiles 的指纹
            new_fp = get_morgan_fingerprint(new_mol)
            if new_fp is None:
                # 如果连分子都解析不了，可以视需求来决定怎么处理：
                # - 直接重试
                # - 当作已达终点
                # - 或者接受该结果
                n_repeat += 1
                if n_repeat >= self.config.max_n_repeat:
                    # 超过重试上限就返回这个action
                    break
                continue

            # 计算与现有子节点的相似度，如果都在阈值以下，则接受
            if not is_too_similar_to_children(new_fp, children_fps, threshold=0.8):
                # 找到一个不相似的 SMILES
                break
            else:
                n_repeat += 1
                # 若超过重试上限，就直接返回这次的 action（或可改为其它处理）
                if n_repeat >= self.config.max_n_repeat:
                    break

        return action, has_end_token, n_repeat

    def can_expand(self):
        """判断当前节点自身是否可扩展（生成新的子节点）"""
        return not self.is_terminal_node() and not self.is_fully_expanded()

    def has_expandable_descendant(self):
        """递归检查当前节点或其子孙节点是否可扩展"""
        if not self.expandable:
            return False
        # 如果当前节点自身可扩展，直接返回 True
        if self.can_expand():
            return True
        # 递归检查所有子节点
        for child in self.children:
            if child.has_expandable_descendant():
                return True
        # 所有子孙节点均不可扩展
        self.expandable = False
        return False

    def best_child(self, alpha=0.5):
        """
        改进的 best_child 逻辑：
        1. 过滤掉没有可扩展后代的子节点
        2. 在剩余子节点中选择 UCT 值最高的
        """
        valid_children = []
        for child in self.children:
            # 只保留有可扩展后代的子节点
            if child.has_expandable_descendant():
                valid_children.append(child)

        # 如果没有有效子节点，返回 None
        if not valid_children:
            return None

        # 计算有效子节点的 UCT 值
        choices_weights = []
        for c in valid_children:
            exploit = alpha * c.q() / c.n() + (1 - alpha) * max(c.result())
            explore = np.sqrt(np.log(self.n()) / c.n())
            uct_value = exploit + self.config.c_param * explore
            choices_weights.append(uct_value)

        # 选择 UCT 值最高的子节点
        idx = np.argmax(choices_weights)
        return valid_children[idx]

    def backpropagate(self, value):
        """
        回溯更新：在本节点累加 result，并递归更新父节点。
        """
        self._number_of_visits += 1
        self._results.append(value)
        if self.parent:
            self.parent.backpropagate(value)

    def _tree_policy(self):
        """
        Select and expand
        Selection strategy: if not fully expanded, pick current node, otherwise pick best child and check
        MCTS 核心：迭代策略
          1. 向下选择(Selection)
          2. 扩展(Expansion)
        """
        current_node = self
        while not current_node.is_terminal_node():  # 只要不是终止结点，就已知向下搜索
            current_node.update_n_total_children(
                self.config.width_increase_factor)  # 根据子节点的情况自适应地增加/减少 n_total_children_adaptive
            if not current_node.is_fully_expanded():  # 如果当前结点还能继续扩展，就继续扩展
                # 扩展一个新的子节点
                return current_node.expand(), True
            else:
                current_node = current_node.best_child()  # 如果当前结点不能继续扩展，就找它的孩子结点
                if current_node is None:
                    return self, False
        return current_node, False

    def add_value(self, is_additional=False):
        """
        用于对当前分子进行一次“价值评估”，如 QED、LogP 或其他性质。
        如果 is_additional=True，可表示评估另一种性质（比如毒性评分）。
        """
        raw_value = self.state.get_value()

        # 如果需要，你也可以对 raw_value 做归一化或缩放，如：
        # raw_value = (raw_value - 0.5) * 2  # 将 [0,1] 区间映射到 [-1,1] 之类
        return raw_value

    def add_simulate(self):
        """
        做一个“快速模拟/评估”。
        在分子GPT场景下，可以:
          1. 随机在当前分子基础上扩展几步
          2. 计算每次得到的分子打分
          3. 取平均或其他统计值
        这样让节点在还未真正完全展开时，就对可能的后续做一个估计，用于指导MCTS。
        """
        value = self.fast_rollout_evaluation()

        # 此处示例返回平均值
        return value

    def fast_rollout_evaluation(self):
        """
        Fast-rollout and return mean value from ORM.
        """

        action, smiles_answer, has_end_token = self.state.generate_fragment(
            cur_molecule=self.state.cur_molecule,
            max_seq_len=1024,
            temperature=1.0,
            top_k=None,
            stream=False,
            rp=1.0,
            kv_cache=True,
            is_simulation=True
        )
        _, smiles = sentence2mol(smiles_answer)
        value = self.state.get_reward(smiles)

        return value

    def update_n_total_children(self, increase_factor):
        """
        如果想根据子节点的情况自适应地增加/减少 n_total_children_adaptive，
        可以在这里自定义逻辑。
        """
        if not self.children:
            return
        # 简单示例：以子节点平均价值为参考做一些扩展
        values = [np.sum(child.q()) / child.n() for child in self.children]
        values = np.array(values)
        mean_value = np.mean(values)
        diff_values = np.abs(values - mean_value)
        value_diff = np.max(diff_values)
        if value_diff > self.max_q_diff:
            self.max_q_diff = value_diff

        new_n_total_children = min(int(increase_factor * value_diff), 10)
        if new_n_total_children > self.n_total_children_adaptive:
            self.n_total_children_adaptive = new_n_total_children

        # 若还有别的规则，比如重复次数、方差判断等，也可以在这里加

    def best_action_global_leaf(self):
        """
        找到整棵子树中自身最大单次 reward 值最高的叶子节点。
        """
        if self.is_terminal_node():
            return self  # 自己就是叶子

        best_leaf = None
        highest_reward = float('-inf')

        for child in self.children:
            leaf = child.best_action_global_leaf()  # 递归查找子树
            if leaf is None:
                continue  # 忽略非终止节点或无效子树
            current_reward = max(leaf.result()) if leaf.result() else 0  # 取该叶子节点的最大单次 reward

            if current_reward > highest_reward:
                highest_reward = current_reward
                best_leaf = leaf

        return best_leaf

    def best_child_greedy(self):
        """
        简单的贪心策略(不加探索项)。
        """
        if not self.children:
            return None
        choices = [c.q() / c.n() if c.n() > 0 else 0 for c in self.children]
        idx = np.argmax(choices)
        return self.children[idx]

    def best_action_greedy_leaf(self):
        """
        递归找到底层节点(贪心)。
        """
        current_node = self
        while not current_node.is_terminal_node():
            next_node = current_node.best_child_greedy()
            if next_node is None:
                break
            current_node = next_node
        return current_node

    def get_end_state(self):
        """
        如果需要“一次性生成到结束”，可以在 state 中写好 take_action_end。
        """
        end_state = self.state.take_action_end(is_greedy=False)
        return end_state

    # 一些用于调试或收集信息的辅助方法：
    def generate_all_paths(self):
        """
        从当前节点遍历所有子树，把路径都返回。
        """
        all_paths = []
        all_path_set = set()
        queue = deque(self.children)
        while queue:
            cur = queue.popleft()
            cur_path = cur.state.cur_molecule
            if cur_path in all_path_set:
                continue
            all_paths.append({
                "path": cur_path,
                "depth": cur.depth,
                "score": cur.get_cached_reward(),
                "is_terminal": cur.is_terminal_node()
            })
            all_path_set.add(cur_path)
            queue.extend(cur.children)
        return all_paths

    def get_all_leaves(self):
        """获取所有叶节点。"""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves


class MCTS:
    def __init__(self, initial_state, config, args=None):
        """
        initial_state:  初始 ProblemState，包含生成分子GPT的上下文等
        config:         包含各类超参的字典，如搜索时间、c_param 等
        args:           可能带有命令行参数等
        """
        self.initial_state = initial_state

        self.config = config
        self.args = args

        self.root = None
        self.max_search_depth = 0
        self.unique_nodes = set()
        self.time_taken = 0

    def run_mcts(self):
        """
        MCTS 主循环：Selection / Expansion / Evaluation / Backpropagation
        """
        # 创建根节点
        if self.root is None:
            self.root = MonteCarloTreeSearchNode(state=self.initial_state,
                                                 config=self.config,
                                                 depth=0,
                                                 node_id='root')

        search_iter = 0
        n_terminals = 0

        n_steps, n_rollouts, n_requests = 0, 0, 0

        # 执行蒙特卡洛树搜索
        pbar = tqdm(range(self.config.search_time),
                    desc="MCTS simulations",
                    leave=True)
        # 初始化记录变量
        buffer = {}
        # 继续搜索，直到达到给定的时间次数 或者找到足够多终止节点
        while search_iter < self.config.search_time or n_terminals < self.config.min_terminals:
            search_iter += 1
            pbar.update(1)
            # 1) selection + expansion
            v, is_expand = self.root._tree_policy()

            # 2) 如果确实扩展了节点，就评估( Evaluation ) + 回溯( Backpropagation )
            if is_expand:
                reward = 0.0
                # 2.1) 根据 value_weight 调用节点的 add_value()
                if self.config.value_weight > 0:
                    # 例如：对分子做一次价值评估(QED)并加权
                    raw_value = v.add_value(is_additional=False)
                    reward += self.config.value_weight * raw_value

                # # 如果还有额外价值网络 (add_value)，也可以同样加和
                # if self.config.add_value_weight > 0:
                #     reward += self.config.add_value_weight * v.add_value(is_additional=True)

                # 2.2) fast-rollout (simulate)：如果想做“快速模拟”或“快速评估”
                if self.config.n_simulations > 0 and self.config.fastrollout_weight > 0:
                    if v.is_terminal_node():
                        # 如果当前节点已经是终止，就可能直接拿到终止打分
                        # (示例) reward += self.config.fastrollout_weight * v.get_final_molecule_outcome()
                        raw_value = v.add_value(is_additional=False)
                        reward += self.config.fastrollout_weight * raw_value
                    else:
                        # 否则做一次快速模拟
                        raw_value = v.add_simulate()
                        reward += self.config.fastrollout_weight * raw_value

                # 缓存这个 reward
                v.set_cached_reward(reward)
                # 回溯更新
                v.backpropagate(reward)

                # 打印或日志记录(可选)
                # if self.args and self.args.debug_log_level >= 3:
                #     print(f"Rollout: {n_rollouts}, Depth: {v.depth}, Reward: {reward:.2f}")

                # 2.4) 统计计数
                # 注意这里 parent_action 是指“从父节点到本节点”的生成内容(可能是SMILES片段)
                parent_action = v.parent_action if v.parent_action else ""
                # 这里示例行数: 用换行来粗略衡量“步骤数量”，可根据需要改成 token 数等
                n_action_steps = parent_action.count(13) - 1  # 第一个[SEP]是给定的
                n_steps += n_action_steps
                n_rollouts += 1
                # 如果状态合并时有重复(类似编辑距离或相似度多次重试)，可以记录到n_requests
                n_requests += v.n_repeat_by_parent * n_action_steps

                if v.is_terminal_node():
                    n_terminals += 1
                    buffer[v.node_id] = [reward, search_iter]

                if search_iter % self.config.freq_log == 0:
                    auc_top10 = top_auc(buffer, 10, self.config.freq_log, self.config.search_time)
                    auc_top100 = top_auc(buffer, 100, self.config.freq_log, self.config.search_time)
                    print(f"\niter.{search_iter} AUC for Best 10 Mean: {auc_top10:.4f}")
                    print(f"iter.{search_iter} AUC for Best 100 Mean: {auc_top100:.4f}")
            else:
                # 如果没扩展，则说明该节点是之前搜索过的，这次只做回溯
                reward = v.get_cached_reward()
                v.backpropagate(reward)

            # 更新搜索深度
            if v.depth > self.max_search_depth:
                self.max_search_depth = v.depth

        # 关闭进度条
        pbar.close()

        auc_top10 = top_auc(buffer, 10, self.config.freq_log, self.config.search_time)
        auc_top100 = top_auc(buffer, 100, self.config.freq_log, self.config.search_time)
        print(f"\niter.{search_iter} AUC for Best 10 Mean: {auc_top10:.4f}")
        print(f"iter.{search_iter} AUC for Best 100 Mean: {auc_top100:.4f}")

        # 更新 MCTS 的累积统计
        self.total_rollouts = n_rollouts
        self.total_steps = n_steps
        self.total_requests = n_requests

        self.save_tree('./tree_log/root_.p')
        # 使用 'w' 模式写入文件，确保指定编码以支持中文
        with open("./tree_log/buffer_.json", "w", encoding="utf-8") as f:
            json.dump(buffer, f, ensure_ascii=False, indent=4)

        return None

    def run(self):
        start_time = time.time()
        self.run_mcts()
        end_time = time.time()
        self.time_taken = end_time - start_time
        print(f"run_time:{self.time_taken / 60 :.2f}min")
        return None

    def get_time(self):
        return self.time_taken

    def get_max_search_depth(self):
        return self.max_search_depth

    # 下面这些函数都是原先用于获取终止节点、路径等，如果需要可保留
    def get_all_paths(self):
        return self.root.generate_all_paths() if self.root else []

    def get_final_state_greedy(self):
        if not self.root:
            return None
        greedy_leaf = self.root.best_action_greedy()
        return greedy_leaf.get_end_state()

    def get_final_state_global(self):
        if not self.root:
            return None
        best_leaf = self.root.best_action_global_leaf()
        return best_leaf.get_end_state()

    # 如果还需要对树做序列化、保存、加载之类，可保留。也可以去掉
    def save_tree(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.root, f)

    @classmethod
    def load_tree(cls, filename, config):
        with open(filename, 'rb') as f:
            root = pickle.load(f)
        # 重建 MCTS
        mcts_recover = cls(initial_state=None, config=config)
        mcts_recover.root = root
        return mcts_recover





