import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
# from simulate_node import simulate, setup, prepare_once
from backend.simulate_efficient_hardhat import simulate, setup, prepare_once
from optimize import get_groundtruth_order, get_params, substitute

import argparse
import sys
import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # 兼容 Python 的 IO flush
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run optimization process.")
    
    # 添加参数 -a 或 --address
    parser.add_argument(
        "-a", "--address",          # 参数名
        type=str,                   # 参数类型
        required=False,              # 是否必填
        default="./default_path",   # 默认值
        help="Path to the address or config file."
    )
    
    # 你还可以加更多参数
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=False, 
        default="./default_path",
        help="Path to save the result."
    )

    parser.add_argument(
        "-p", "--port",
        type=int,
        required=True, 
        default=8601,
        help="Port."
    )
    
    return parser.parse_args()

class ParameterPolicyNetwork(nn.Module):

    def __init__(self, input_size, param_size, learning_rate=0.001, exploration_weight=0.1,entropy_weight=1e-3):
        super(ParameterPolicyNetwork, self).__init__()

        # 第一部分：卷积层，提取局部特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # 全连接层
        conv_output_size = input_size * 32  # 假设输入为1D数据，卷积后的输出大小为 input_size * 32
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, param_size)  # 输出参数

        # 归一化层
        self.bn_fc1 = nn.LayerNorm(128)
        self.bn_fc2 = nn.LayerNorm(64)

        # 残差连接
        self.residual_fc = nn.Linear(conv_output_size, 128)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(p=0.5)

        # 激活函数
        self.relu = nn.ReLU()

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function = self._loss
        self.exploration_weight = exploration_weight

    def forward(self, state):
    
        # 数据归一化
        state = (state - state.mean()) / (state.std() + 1e-8)
        state = state.unsqueeze(1)  # 扩展维度用于卷积层

        # 卷积层 + 批归一化 + 激活函数
        x = self.conv1(state)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # 将卷积层的输出展平
        x = x.view(x.size(0), -1)

        # 残差连接
        residual = self.residual_fc(x)
        x = self.fc1(x) + residual  # 残差连接
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 全连接层 + 批归一化 + 激活函数
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 输出层
        # param_values = F.relu(self.fc3(x))  # 保证输出非负
        param_values = torch.sigmoid(self.fc3(x))
        # param_values = torch.sigmoid(self.fc3(x)) + 0.05 * torch.randn_like(x)
        # param_values = torch.clamp(param_values, 0.0, 1.0)
        return param_values

    def predict(self, state):
        # 输入当前交易序列并进行预测，返回参数值
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为tensor
        with torch.no_grad():
            param_values = self.forward(state_tensor).squeeze(0)
        return param_values.numpy()

    def update(self, sequences, rewards_batch):
        # 将参数和奖励用于策略更新
        self.optimizer.zero_grad()  # 清除梯度

        losses = []  # 存储每个参数的损失
        
        for sequence, reward in zip(sequences, rewards_batch):
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # 转换为tensor
            action_probs = self.forward(sequence_tensor).squeeze(0)
            loss = self.loss_function(action_probs, reward)
            losses.append(loss)

        # 将所有的损失求和，进行反向传播
        total_loss = torch.stack(losses).mean()  # 计算平均损失
        total_loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新网络权重

    def _loss(self, param_values, reward):
        # 损失函数为负对数似然损失函数，乘以奖励来引导策略
        log_probs = torch.log(param_values + 1e-8)  # 防止log(0)
        return -log_probs.sum() * reward  # 使用奖励来引导损失


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, learning_rate=0.001, exploration_weight=0.1, num_heads=4):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_size)  # 输出与输入大小一致，每个交易的概率
        
        # Attention 机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=0.1)
        
        # 增加残差连接
        self.fc1_residual = nn.Linear(input_size, 128)
        self.fc2_residual = nn.Linear(128, 64)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function = self._loss
        self.exploration_weight = exploration_weight

    def forward(self, state):
        # 数据归一化
        state = (state - state.mean()) / (state.std() + 1e-8)
        
        # 全连接层1 + 残差连接
        x = torch.relu(self.fc1(state)) + torch.relu(self.fc1_residual(state))
        x = self.layer_norm1(x)  # Layer Norm
        x = self.dropout(x)
        
        # 全连接层2 + 残差连接
        x = torch.relu(self.fc2(x)) + torch.relu(self.fc2_residual(x))
        x = self.layer_norm2(x)  # Layer Norm
        
        # 注意力机制
        x = x.unsqueeze(0)  # MultiheadAttention 需要输入为 (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)

        # 输出层
        action_probs = self.softmax(self.fc3(attn_output))
        return action_probs

    def predict(self, state):
        # 输入当前状态并进行预测，返回动作的概率分布
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为tensor
        with torch.no_grad():
            action_probs = self.forward(state_tensor).squeeze(0)
        return action_probs.numpy()

    def update(self, state_action_pairs, rewards):
        # 将序列和奖励用于策略更新
        self.optimizer.zero_grad()  # 清除梯度
        total_loss = 0.0

        (state, action) = state_action_pairs
        reward = rewards

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor).squeeze(0)

        # 防止 log(0)
        log_prob = torch.log(action_probs[action] + 1e-8)
        loss = -log_prob * reward  # Policy gradient
        total_loss += loss

        # 加上熵正则项以保持探索
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        total_loss -= self.exploration_weight * entropy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # 防梯度爆炸
        self.optimizer.step()

        # # 将所有的序列转换为tensor
        # losses = []  # 存储每个序列的损失

        # for sequence, reward in zip(sequences, rewards):
        #     sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # 转换为tensor
        #     action_probs = self.forward(sequence_tensor).squeeze(0)
        #     loss = self.loss_function(action_probs, reward)
        #     losses.append(loss)

        # # 将所有的损失求和，进行反向传播
        # total_loss = torch.stack(losses).mean()  # 计算平均损失
        # total_loss.backward()  # 计算梯度
        # self.optimizer.step()  # 更新网络权重

    def _loss(self, action_probs, reward):
        # 损失函数为负对数似然损失函数，乘以奖励来引导策略
        log_probs = torch.log(action_probs)
        entropy = -torch.sum(action_probs * log_probs)  # 计算熵
        return -log_probs.sum() * reward - self.exploration_weight * entropy  # 加上熵作为探索的一部分

class MCTS_DRLSolver:
    def __init__(self, transaction_pool, policy_network, param_policy_network, num_simulations=100, exploration_weight=10, max_iterations=1,):
        self.transaction_pool = transaction_pool
        self.num_actions = len(transaction_pool)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.policy_network = policy_network
        self.param_policy_network = param_policy_network
        self.tree = {}  # 存储蒙特卡罗树的节点
        self.max_iterations = max_iterations  # 控制最多搜索轮数

    def search(self, state):
        for _ in range(self.num_simulations):
            self.simulate(state)  # 添加初始深度为0

        bestaction = self.best_action(state)
        reward = self.evaluate_DRL(state, bestaction)
        # result2 = [int(x) for x in result]
        
        return self.best_action(state),reward

    def simulate(self, state):  
        # 显式使用栈来模拟递归行为
        stack = [(state, 0, None)]  # 初始化栈，存储 (当前状态, 当前奖励, 上一个选择的动作)
        path = []  # 记录访问过的路径
        cumulative_reward = 0
        # print(stack, path, cumulative_reward)

        while stack:

            # print(stack, path, cumulative_reward)

            current_state, reward, action_taken = stack.pop()  # 从栈中取出状态、奖励和所采取的动作
            

            state_tuple = tuple(current_state)

            if state_tuple not in self.tree:
                # 叶节点, 使用策略网络初始化
                action_probs = self.policy_network.predict(current_state)
                # print("Action probs:", action_probs)
                self.tree[state_tuple] = {
                    "N": [0] * self.num_actions,  # 每个动作的访问次数初始化为0
                    "W": [0.0] * self.num_actions,  # 每个动作的累计奖励初始化为0.0
                    "Q": [0.0] * self.num_actions,  # 每个动作的平均奖励初始化为0.0
                    "P": action_probs  # 策略网络预测的动作概率
                }
                # cumulative_reward += reward  # 叶节点累加奖励
                # if self.is_terminal(current_state):
                #     return cumulative_reward  # 如果是终止状态，直接返回累计奖励
                break

            node = self.tree[state_tuple]
            
            # 使用 UCB1 算法选择动作
            action = self.select_action(current_state, node["P"])
            if action == -1:
                return cumulative_reward


            next_state = self.next_state(current_state, action)

            # Get reward based on policy network's confidence for the chosen action
            reward = self.evaluate_DRL(current_state, action)  # 可自定义评估函数

            path.append((node, action, reward))

             # 在路径上每一步都累加奖励
            cumulative_reward += reward  # 在每一步更新累积奖励

            # 将下一个状态和动作、奖励入栈
            stack.append((next_state, reward, action))

        R = 0
        gamma = 1.0  # or 0.9 if you want discounted return

        # 回溯更新路径中的所有节点
        for node, action_taken, reward in reversed(path):

            R = reward + gamma * R

            # 更新被选动作的统计数据
            node["N"][action_taken] += 1  # 该动作的访问次数增加

            # node["W"][action_taken] += cumulative_reward  # 累计奖励增加
            node["W"][action_taken] += R # 使用回溯时的累计奖励更新 W 值代替【cumulative_reward】

            node["Q"][action_taken] = node["W"][action_taken] / node["N"][action_taken]  # 更新平均奖励\
            # 在回溯过程中，累积奖励仍然保持不变，确保奖励正确传播到每个节点
            #cumulative_reward += reward  # 将当前回溯的奖励累加到累计奖励中

        return cumulative_reward

    def is_terminal(self, state):
        # 检查是否为终止状态
        # 例如，如果所有交易都已执行，则返回 True
        return sum(state) >= len(state)  # 如果所有交易均已执行

    def select_action(self, state, action_probs):
    # 获取当前节点的统计信息
        node = self.tree[tuple(state)]
        
        # 计算当前节点所有动作的访问次数总和
        total_N = sum(node["N"])  # 只需当前节点的 N 值

        # UCB1: Q + exploration_weight * P * sqrt(total_N) / (1 + N)
        ucb_values = [
            node["Q"][i] + self.exploration_weight * action_probs[i] * np.sqrt(total_N + 1) / (1 + node["N"][i])
            for i in range(len(action_probs))  # 根据 action_probs 长度来遍历动作
        ]

        # 排除已经被选择过的非法动作（状态中为 1 的动作）
        valid_actions = [i for i in range(len(action_probs)) if tuple(state)[i] != 1]

        # 如果合法动作为空，返回一个默认值（比如随机返回）
        if not valid_actions:
            return -1
        
        # if sum(node["N"]) == 0: return valid_actions[np.argmax([node["P"][i] for i in valid_actions])]  # 如果所有动作都未被访问过，选择概率最高的动作
        if sum(node["N"]) == 0:
            # 从未访问过的节点，用先验概率P进行采样
            valid_P = np.array([node["P"][i] for i in valid_actions], dtype=np.float64)
            valid_P = np.maximum(valid_P, 1e-8)  # 避免全为0
            valid_P = valid_P / np.sum(valid_P)  # 归一化成概率分布
            # 可选：根节点添加Dirichlet噪声，增强探索性
            if tuple(state) == tuple([0] * len(state)):  # 根节点
                epsilon = 0.25
                dirichlet_alpha = 0.3
                noise = np.random.dirichlet([dirichlet_alpha] * len(valid_actions))
                valid_P = (1 - epsilon) * valid_P + epsilon * noise
                valid_P = valid_P / np.sum(valid_P)

            return np.random.choice(valid_actions, p=valid_P)

            #return np.random.choice(valid_actions, p=valid_P)


        # 当所有动作的 UCB 值都很接近时，增加探索性，使用概率选择
        max_ucb_value = max([ucb_values[i] for i in valid_actions])
        min_ucb_value = min([ucb_values[i] for i in valid_actions])

        # 如果最大值和最小值之间的差距很小，说明UCB1没有区分度，则使用概率选择
        if max_ucb_value - min_ucb_value < 1e-9:
        # 使用策略网络的 action_probs 进行采样，增加随机性
            return np.random.choice(valid_actions)

        # 筛选合法动作的最大 UCB 值
        best_action = valid_actions[np.argmax([ucb_values[i] for i in valid_actions])]

        return best_action


    def next_state(self, state, action):
        # 返回选择action后新的状态
        next_state = state.copy()
        next_state[action] = 1  # 假设动作是将交易状态设置为已执行
        return next_state

    def best_action(self, state):
        # 在搜索结束后，选择访问次数最多的动作作为最终动作
        node = self.tree[tuple(state)]
        # print(node)
        # 选择访问次数最多的动作
        best_action_index = np.argmax(node["N"])  # 选择访问次数最大的动作
        return best_action_index

    def update_policy_network(self, state_action_pairs, rewards):
        # 使用最优序列和MEV奖励更新策略网络
        self.policy_network.update(state_action_pairs, rewards)

    def update_parameter_network(self, best_params, rewards):
        # 假设 reward 是一个标量，代表当前最优序列的MEV奖励
        self.param_policy_network.update(best_params, rewards)

    # def run(self, initial_state):
    #     # 运行MCTS搜索并通过策略网络进行更新
    #     best_sequence = []
    #     state = initial_state
    #     while sum(state) < len(state):  # 直到所有交易都执行
    #         action = self.search(state)
    #         best_sequence.append(action)
    #         state = self.next_state(state, action)

    #     # 计算当前最优序列的MEV奖励
    #     reward = self.evaluate(best_sequence)
    #     self.update_policy_network(best_sequence, reward)
    #     return best_sequence, reward
    
    def run(self, initial_state, batch_size=1):
        best_sequence_overall = []
        best_reward_overall = float('-inf')
        
        for iteration in range(self.max_iterations):
            print("============== Iteration", iteration, "===============")
            print("best_sequence: ",[int(x) for x in best_sequence_overall])
            print("best_MEV: ",best_reward_overall)
            print("============== Iteration", iteration, "===============")
            best_sequences = []
            mevrewards = []
            params = []
            for _ in range(batch_size):
                best_sequence = []
                mcts_collected_trajectories = []
                state = initial_state
                self.tree = {}  # 每次重新开始搜索时重置搜索树

                start = time.time()

                # 执行搜索，找到当前最优序列
                while not self.is_terminal(state):
                    action,action_reward = self.search(state)
                    if action is None:
                        break
                    best_sequence.append(action)
                    mcts_collected_trajectories.append([(state.copy(), action),action_reward])
                    state = self.next_state(state, action)

                result = [int(x) for x in best_sequence]

                print("Search sequence: ",result)

                end = time.time()
                print(f"MCTS搜索用时: {end - start:.4f} 秒")

                # (best_sequence)

                start = time.time()

                param = self.param_policy_network.predict(best_sequence)

                end = time.time()
                print(f"参数预测 param_policy_network 用时: {end - start:.4f} 秒")

                params_order = get_params(transactions)
                # print(params_order)
                # print(domain)
                print(param)
                # print([domain[name][1] for name in params_order])

                # param = param * [domain[name][1] for name in params_order]
                param = [p * domain[name][1] for p, name in zip(param, params_order)]

                param_show = [float(x) for x in param]
                print(param_show)

                start = time.time()

                reward = self.evaluate(best_sequence, param)

                end = time.time()
                print(f"MEV Evaluation用时: {end - start:.4f} 秒")

                params.append(param)
                best_sequences.append(best_sequence)
                mevrewards.append(reward)
                print("")
                print("=====================================================")
                print("Iteration:", iteration, "\nSequence:", result, "\nReward:", reward)
                print("=====================================================")
                print("")

            # start = time.time()

            # 更新策略网络
            #self.update_policy_network(best_sequences, rewards)
            # 更新时，不用完整 sequence，而是每次 MCTS simulation 的 state-action 对
            for state_action_pairs, rewardss in mcts_collected_trajectories:
                self.update_policy_network(state_action_pairs, rewardss)

            for _ in range(len(initial_state)):
                self.update_parameter_network(best_sequences, mevrewards)

            # end = time.time()
            # print(f"Update 2 networks 更新用时: {end - start:.4f} 秒")

            # 比较当前序列的奖励与全局最优序列
            for i in range(len(best_sequences)):
                if mevrewards[i] > best_reward_overall:
                    best_reward_overall = mevrewards[i]
                    best_sequence_overall = best_sequences[i]

        return best_sequence_overall, best_reward_overall
    
    def evaluate_DRL(self, state, action):
        # print(state)
        # 使用策略网络预测当前状态的动作概率
        action_probs = self.policy_network.predict(state)   
        # 1. 选择的动作的置信度得分
        action_confidence = action_probs[action] * 100
        # print(action_confidence)   
        # 2. 动作概率的熵值，反映网络的不确定性（鼓励探索）
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))  # 小的epsilon防止log(0)的出现
        # print(entropy) 
        # 3. 基于状态的启发式奖励：鼓励朝完成交易池的方向前进
        progress = np.sum(state) / self.num_actions  # 已完成的交易数占总交易数的比例
        # print(progress)  
        # 4. 时间衰减：越早采取的动作奖励越高
        discount_factor = 0.9  # 可调参数，应用时间衰减
        step = np.sum(state)  # 基于已采取的动作数量来确定当前的步数
        time_discount = discount_factor ** step  # 随着序列的推进，奖励逐步衰减
        alpha = 0.2  # 控制熵对奖励的影响程度
        # 将这些因素组合成最终的奖励
        reward = action_confidence * (1 - alpha * entropy) * (1 + progress) * time_discount
        # print(reward)
        return reward

    def evaluate(self, sequence, sample):
        
        temp_transactions = reorder(transactions, sequence)
        params = get_params(transactions)

        sample_dict = {}
        for p_name, v in zip(params, sample):
            sample_dict[p_name] = v * domain_scales[p_name]

        datum = substitute(temp_transactions, sample_dict, cast_to_int=True)

        mev = simulate(ctx, datum, port_id, involved_dexes, False, '', 'max')
        
        mev = mev#* random.randint(1,10)
        # 假设一个MEV评估函数，这里返回一个简单的模拟MEV奖励
        return mev  # 示例：返回一个随机奖励


def reorder(transactions, order):
    '''
        function to reorder a set of transactions, except for the first one
    '''
    # order = order.astype(np.int32)
    reordered_transactions = [transactions[0]] + [transactions[i+1] for i in order]
    return reordered_transactions


args = parse_args()

result_path = args.output

folder_path = args.address

dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

for file in dirs:

    transactions_addr = os.path.join(os.path.join(folder_path, file),"amm_reduced")
    domain_addr = os.path.join(os.path.join(folder_path, file),"domain")

    os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_path,f"run_log_{file}.txt")

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print(f"[*] Log started at {timestamp}")
        
    # 打印看看
    print(f"Address: {transactions_addr}")
    print(f"Output: {os.path.join(result_path,file)}")


    # transactions_addr = "./backend/aave2"
    # domain_addr = "./backend/aave2_domain"

    port_id = args.port

    transactions_f = open(transactions_addr, 'r')
    domains_f = open(domain_addr, 'r')

    domain = {}
    domain_scales = {}
    new_domain = domain_addr
    VALID_RANGE = {}

    for line in domains_f.readlines():
        if line[0] == '#':
            continue
        tokens = line.strip().split(',')
        # print('=>', tokens)
        lower_lim, upper_lim = float(tokens[1]), float(tokens[2])
        token_pair = new_domain.split('/')[-2]
        # print(token_pair) 
        if len(tokens)==3:
            VALID_RANGE[token_pair] = min(1e6, upper_lim)
            if upper_lim > VALID_RANGE[token_pair]:
                domain_scales[tokens[0]] = upper_lim / VALID_RANGE[token_pair]
                upper_lim = VALID_RANGE[token_pair]
            else:
                domain_scales[tokens[0]] = 1.0
        else:
            assert len(tokens)==4
            domain_scales[tokens[0]] = float(tokens[3])
        domain[tokens[0]] = (lower_lim, upper_lim)

    print(domain)
    print(domain_scales)

    transactions = transactions_f.readlines()
    # print(transactions)
    print("Setting up...", transactions[0])
    ctx = setup(transactions[0], capital=10000)

    involved_dexes = ['sushiswap', 'aave', 'uniswapv3']

    print("DEXs: ", involved_dexes)

    ctx = prepare_once(ctx, transactions, port_id, involved_dexes)
    print("Prices: ", ctx.prices)

    ################ test ################ 

    # param = [0.1, 0.5, 0.3, 0.7, 0.2]
    # params_order = get_params(transactions)
    # # print(params_order)
    # # print(domain)
    # print(param)
    # print(params_order)
    # # print([domain[name][1] for name in params_order])

    # param = [p * domain[name][1] for p, name in zip(param, params_order)]
    # print(param)

    # sequence = [ 1 , 9 ,20 ,12 , 3 ,16, 10, 11  ,6 ,2 ,15, 13 , 7 , 0 ,19, 18, 14,  8 ,17,  4 , 5 ]

    # # gt_order = get_groundtruth_order(transactions[1:], include_miner=True)
    # # print('groundtruth order:', gt_order)

    # print("simulating...")
    # # time.sleep(10000)

    # temp_transactions = reorder(transactions, sequence)
    # params = get_params(transactions)

    # sample_dict = {}
    # for p_name, v in zip(params, param):
    #     sample_dict[p_name] = v * domain_scales[p_name]

    # datum = substitute(temp_transactions, sample_dict, cast_to_int=True)
    # # print(datum)
    # #time.sleep(10000)

    # mev = simulate(ctx, datum, port_id, involved_dexes, False, '', 'max')
    # print(mev)

    # time.sleep(10000)

    ################ test ################ 

    # 假设有一个包含5个交易的交易池
    # transaction_pool = ['T1', 'T2', 'T3', 'T4', 'T5']
    input_size = len(transactions[1:])

    print("Input size: ", input_size)

    print("")

    # 创建策略网络
    policy_network = PolicyNetwork(input_size=input_size)
    param_policy_network = ParameterPolicyNetwork(input_size=input_size, param_size=len(domain))

    # 创建MCTS_DRL solver
    solver = MCTS_DRLSolver(transactions[1:], policy_network, param_policy_network)

    # 定义初始状态 (假设所有交易尚未执行)
    initial_state = [0] * input_size

    # print(initial_state)

    # 运行MCTS + DRL流程，查找最优交易序列
    best_sequence, reward = solver.run(initial_state)

    print("Best sequence:", [int(x) for x in best_sequence])
    print("Best Reward:", reward)

    with open(os.path.join(result_path,f"result.txt"), "a", encoding="utf-8") as f:
        f.write(f"File: {file}\n")
        f.write(f"Best sequence: {[int(x) for x in best_sequence]}\n")
        f.write(f"Best Reward: {reward}\n")

