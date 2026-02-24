import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
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
        self.terminal.flush()
        self.log.flush()


# ============================================================
# 奖励归一化器
# ============================================================
class RewardNormalizer(object):
    """
    奖励归一化器：使用 Welford 在线算法维护运行均值和方差。
    用于计算 REINFORCE 的 advantage = (reward - baseline) / std，
    降低策略梯度的方差，稳定训练过程。

    放置位置：
    - update() 在每次获得新的真实 MEV 奖励时调用（run() 中 evaluate 之后）
    - normalize() 在从 replay buffer 采样后、计算 advantage 时调用
    - 仅用于 GaussianParameterNetwork 的 REINFORCE 更新
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Welford 在线方差

    def update(self, value):
        """用新的奖励值更新运行统计量（仅在获得新奖励时调用）"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        return np.sqrt(self.M2 / (self.count - 1)) + 1e-8

    def normalize(self, value):
        """使用当前统计量归一化（不更新统计量），返回 advantage"""
        return (value - self.mean) / self.std


# ============================================================
# 经验回放缓冲区
# ============================================================
class ReplayBuffer(object):
    """
    经验回放缓冲区：存储历史搜索经验用于离线网络更新，提高样本效率。
    分为两个独立缓冲区：
    - policy_value_buffer: 存储 (state, mcts_probs, raw_mev)，用于 PolicyNetwork 和 ValueNetwork
    - param_buffer: 存储 (sequence, param_action, raw_mev)，用于 GaussianParameterNetwork
    """
    def __init__(self, capacity=5000):
        self.policy_value_buffer = deque(maxlen=capacity)
        self.param_buffer = deque(maxlen=capacity)

    def push_policy_value(self, state, mcts_probs, raw_mev):
        """存储 (状态, MCTS搜索分布, 真实MEV)"""
        self.policy_value_buffer.append((
            list(state), list(mcts_probs), float(raw_mev)
        ))

    def push_param(self, sequence, param_action, raw_mev):
        """存储 (排序序列, 采样的参数值, 真实MEV)"""
        self.param_buffer.append((
            list(sequence), list(param_action), float(raw_mev)
        ))

    def sample_policy_value(self, batch_size):
        k = min(batch_size, len(self.policy_value_buffer))
        return random.sample(list(self.policy_value_buffer), k) if k > 0 else []

    def sample_param(self, batch_size):
        k = min(batch_size, len(self.param_buffer))
        return random.sample(list(self.param_buffer), k) if k > 0 else []

    def __len__(self):
        return len(self.policy_value_buffer)


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MCTS+RL MEV Optimization")

    parser.add_argument("-a", "--address", type=str, required=False,
                        default="./default_path", help="数据目录路径")
    parser.add_argument("-o", "--output", type=str, required=False,
                        default="./default_path", help="结果保存路径")
    parser.add_argument("-p", "--port", type=int, required=True,
                        default=8601, help="Hardhat 端口号")

    # MCTS 参数
    parser.add_argument("--num_simulations", type=int, default=100,
                        help="每步 MCTS 模拟次数 (default: 100)")
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="外层迭代轮数 (default: 5)")
    parser.add_argument("--exploration_weight", type=float, default=10.0,
                        help="PUCT 探索权重 c (default: 10.0)")
    parser.add_argument("--rollout_ratio", type=float, default=0.05,
                        help="MCTS 叶节点真实 rollout 比例 (default: 0.05)")

    # 温度参数
    parser.add_argument("--initial_temperature", type=float, default=1.0,
                        help="初始温度 τ (default: 1.0)")
    parser.add_argument("--final_temperature", type=float, default=0.1,
                        help="最终温度 τ (default: 0.1)")

    # 经验回放
    parser.add_argument("--replay_capacity", type=int, default=5000,
                        help="经验回放缓冲区容量 (default: 5000)")
    parser.add_argument("--replay_batch_size", type=int, default=64,
                        help="经验回放采样批量大小 (default: 64)")

    # 网络学习率
    parser.add_argument("--lr_policy", type=float, default=0.001,
                        help="PolicyNetwork 学习率 (default: 0.001)")
    parser.add_argument("--lr_value", type=float, default=0.001,
                        help="ValueNetwork 学习率 (default: 0.001)")
    parser.add_argument("--lr_param", type=float, default=0.001,
                        help="GaussianParameterNetwork 学习率 (default: 0.001)")

    return parser.parse_args()


# ============================================================
# 价值网络（独立设计）
# ============================================================
class ValueNetwork(nn.Module):
    """
    独立价值网络：预测给定交易排序状态（二进制向量）下可达到的预期 MEV。

    架构设计（与 PolicyNetwork 的 Attention 架构有意区分）：
    - 深度残差 MLP：多层全连接 + 跳跃连接，确保梯度畅通
    - LayerNorm 稳定训练
    - 从二进制状态到标量 MEV 值的非线性映射

    训练方式：MSE 损失，目标为真实 simulate 返回的 MEV 值。
    """
    def __init__(self, input_size, hidden_size=128, learning_rate=0.001):
        super(ValueNetwork, self).__init__()

        # 输入投影 + 跳跃连接
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_skip = nn.Linear(input_size, hidden_size)
        self.ln_input = nn.LayerNorm(hidden_size)

        # 残差块 1
        self.res1_fc1 = nn.Linear(hidden_size, hidden_size)
        self.res1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        # 残差块 2
        self.res2_fc1 = nn.Linear(hidden_size, hidden_size)
        self.res2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # 输出头
        self.out_fc = nn.Linear(hidden_size, 64)
        self.out_val = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)

    def forward(self, state):
        # 输入投影 + 跳跃连接
        x = self.relu(self.input_proj(state)) + self.input_skip(state)
        x = self.ln_input(x)

        # 残差块 1
        residual = x
        x = self.relu(self.res1_fc1(x))
        x = self.res1_fc2(x)
        x = self.ln1(x + residual)
        x = self.relu(x)

        # 残差块 2
        residual = x
        x = self.relu(self.res2_fc1(x))
        x = self.res2_fc2(x)
        x = self.ln2(x + residual)
        x = self.relu(x)

        # 输出
        x = self.relu(self.out_fc(x))
        value = self.out_val(x)
        return value

    def predict(self, state):
        """预测状态价值（eval 模式关闭 Dropout 等随机行为）"""
        self.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_tensor).item()
        self.train()
        return value

    def update_batch(self, states, target_values):
        """批量更新：MSE 损失，目标为真实 MEV"""
        self.train()
        self.optimizer.zero_grad()

        total_loss = 0
        for state, target in zip(states, target_values):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            pred = self.forward(state_tensor).squeeze()
            target_tensor = torch.tensor(target, dtype=torch.float32)
            loss = F.mse_loss(pred, target_tensor)
            total_loss += loss

        total_loss /= len(states)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()


# ============================================================
# 策略网络（训练目标改为 MCTS 概率分布）
# ============================================================
class PolicyNetwork(nn.Module):
    """
    策略网络：输出每个交易被选为下一步的概率分布。

    架构：FC + 残差连接 + Multi-Head Attention + Softmax
    改进 (Fix 2): 
    - 保留原论文的 "ResNet + Attention" 架构
    - 将 Attention 改进为 "Feature-level Attention" (特征组注意力)
    - 将 128 维特征重塑为 (8组, 16维)，在特征组之间计算注意力，
      这样既保留了原架构描述，又让 Attention 机制真正发挥作用（捕捉特征间的隐式关联）。
    """
    def __init__(self, input_size, learning_rate=0.001, exploration_weight=0.01, num_heads=4):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_size)

        # Attention 机制 (Feature-level)
        # 将 128 维特征向量重塑为 8 个 "特征头" (Feature Heads) 或 "特征子空间" (Feature Subspaces)。
        # 解释：类似于 Multi-Head Attention 中的多头概念，我们将单一的特征向量解耦为多个独立的语义组。
        # 每一组 (16维) 可能代表交易序列的不同潜在属性（例如：一组关注价格波动，一组关注交易密集度，一组关注Gas费模式等）。
        # Attention 机制计算这些属性之间的动态关联，实现特征层面的自适应加权。
        self.feature_groups = 8
        self.feature_dim = 16  # 128 / 8 = 16
        self.attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, dropout=0.1)

        # 残差连接
        self.fc1_residual = nn.Linear(input_size, 128)
        self.fc2_residual = nn.Linear(128, 64)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.3)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.exploration_weight = exploration_weight  # 熵正则化系数

    def forward(self, state):
        # Fix 3: 移除有害的归一化，保留原始 0/1 状态
        # state = (state - state.mean()) / (state.std() + 1e-8)

        # 全连接层1 + 残差连接
        x = torch.relu(self.fc1(state)) + torch.relu(self.fc1_residual(state))
        x = self.layer_norm1(x)
        x = self.dropout(x)

        # 全连接层2 + 残差连接 (Before Attention)
        # 注意：这里我们调整顺序，在 128 维特征上做 Attention，然后再降维到 64
        
        # Feature-level Attention
        # x shape: (Batch, 128) -> (Batch, 8, 16)
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.feature_groups, self.feature_dim)
        
        # Permute for MultiheadAttention: (Seq_Len, Batch, Embed_Dim) -> (8, Batch, 16)
        x_attn_in = x_reshaped.permute(1, 0, 2)
        
        # Self-Attention between feature groups
        attn_output, _ = self.attention(x_attn_in, x_attn_in, x_attn_in)
        
        # Reshape back: (8, Batch, 16) -> (Batch, 8, 16) -> (Batch, 128)
        x_attn_out = attn_output.permute(1, 0, 2).reshape(batch_size, 128)
        
        # Residual connection around Attention (Optional but good for gradients)
        x = x + x_attn_out
        
        # 继续后续层：128 -> 64
        x = torch.relu(self.fc2(x)) + torch.relu(self.fc2_residual(x))
        x = self.layer_norm2(x)

        # 输出层
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def predict(self, state):
        """预测动作概率分布（Bug3 fix：切换 eval 模式关闭 Dropout）"""
        self.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.forward(state_tensor).squeeze(0)
        self.train()
        return action_probs.numpy()

    def update_with_mcts_probs(self, states, mcts_probs_list):
        """
        用 MCTS 搜索分布作为训练目标（交叉熵损失 + 熵正则化）。
        Loss = -Σ π_MCTS(a) · log π_θ(a) - λ · H(π_θ)

        替代原来的 REINFORCE 更新，优势：
        1. MCTS 分布经过搜索改进，比单步策略更好
        2. 交叉熵损失比 REINFORCE 方差更低
        3. 避免了 evaluate_DRL 的自我强化偏差
        """
        self.train()
        self.optimizer.zero_grad()

        total_loss = 0
        for state, target_probs in zip(states, mcts_probs_list):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            pred_probs = self.forward(state_tensor).squeeze(0)
            target = torch.FloatTensor(target_probs)

            # 交叉熵损失
            policy_loss = -torch.sum(target * torch.log(pred_probs + 1e-8))

            # 熵正则化（鼓励探索，防止策略过早坍缩）
            entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8))

            loss = policy_loss - self.exploration_weight * entropy
            total_loss += loss

        total_loss /= len(states)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()


# ============================================================
# 高斯策略参数网络（替代原 ParameterPolicyNetwork）
# ============================================================
class GaussianParameterNetwork(nn.Module):
    """
    高斯策略参数网络：输出每个参数的均值 μ 和标准差 σ，
    通过从 N(μ, σ²) 采样获得参数值。

    解决原 ParameterPolicyNetwork 的根本问题：
    1. 原网络把 sigmoid 输出当概率用 -log(p)·R 损失 → 所有参数统一推向 0 或 1
    2. 高斯策略提供正确的连续动作空间探索机制
    3. log π(a|s) 正确计算为高斯分布的 log probability

    架构：Conv1D 特征提取 → 共享全连接 → 双头输出 (μ, log σ)
    训练：REINFORCE with Gaussian log probability + 熵正则化
    """
    def __init__(self, input_size, param_size, learning_rate=0.001):
        super(GaussianParameterNetwork, self).__init__()

        # Conv1D 特征提取
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        conv_output_size = input_size * 32

        # 共享全连接层
        self.shared_fc1 = nn.Linear(conv_output_size, 128)
        self.shared_ln1 = nn.LayerNorm(128)
        self.shared_fc2 = nn.Linear(128, 64)
        self.shared_ln2 = nn.LayerNorm(64)

        # 残差连接
        self.residual_fc = nn.Linear(conv_output_size, 128)

        # 均值头（sigmoid → [0, 1]）
        self.mean_head = nn.Linear(64, param_size)

        # log 标准差头（clamp 到合理范围 → σ ∈ [~0.007, ~0.6]）
        self.log_std_head = nn.Linear(64, param_size)

        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)

    def forward(self, state):
        # Fix 3: 移除有害的归一化，保留原始 ID 序列
        state = state.unsqueeze(1)  # (batch, 1, seq_len) for Conv1d

        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten

        residual = self.residual_fc(x)
        x = self.relu(self.shared_fc1(x)) + residual
        x = self.shared_ln1(x)

        x = self.relu(self.shared_fc2(x))
        x = self.shared_ln2(x)

        mean = torch.sigmoid(self.mean_head(x))  # μ ∈ [0, 1]
        log_std = torch.clamp(self.log_std_head(x), -5.0, -0.5)  # σ ∈ [~0.007, ~0.6]

        return mean, log_std

    def predict(self, sequence, deterministic=False):
        """
        预测参数值。
        deterministic=True: 返回均值（用于评估/rollout）
        deterministic=False: 从高斯分布采样（用于探索）
        返回: (action, mean, log_std) 均为 numpy array
        """
        self.eval()
        state_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state_tensor)
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)

            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                action = mean + std * torch.randn_like(mean)
                action = torch.clamp(action, 0.0, 1.0)
        self.train()
        return action.numpy(), mean.numpy(), log_std.numpy()

    def update_batch(self, sequences, actions_taken, advantages):
        """
        REINFORCE 更新：使用高斯分布的 log probability。
        Loss = -Σ log N(a; μ, σ) · advantage - λ · H

        advantage 由 RewardNormalizer 提供:
            advantage = (reward - running_mean) / running_std
        """
        self.train()
        self.optimizer.zero_grad()

        total_loss = 0
        for seq, action, advantage in zip(sequences, actions_taken, advantages):
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
            action_tensor = torch.FloatTensor(action)

            mean, log_std = self.forward(seq_tensor)
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)
            std = torch.exp(log_std)

            # 高斯分布 log probability
            log_prob = -0.5 * (
                ((action_tensor - mean) / (std + 1e-8)) ** 2
                + 2 * log_std
                + np.log(2 * np.pi)
            )
            log_prob = log_prob.sum()

            # REINFORCE 损失: -log π(a|s) · advantage
            policy_loss = -log_prob * advantage

            # 高斯熵正则化（鼓励探索）: H = 0.5 * (1 + log(2πσ²))
            entropy = 0.5 * (1 + 2 * log_std + np.log(2 * np.pi)).sum()

            total_loss += policy_loss - 0.01 * entropy

        total_loss /= len(sequences)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()


# ============================================================
# MCTS + DRL 联合求解器
# ============================================================
class MCTS_DRLSolver:
    """
    MCTS + DRL 联合求解器：通过蒙特卡洛树搜索寻找最优交易排序，
    结合深度强化学习网络持续改进搜索质量。

    核心改进（相比原版本）：
    1. ValueNetwork 独立评估叶节点，替代 evaluate_DRL 的自我置信度（消除自我强化偏差）
    2. PolicyNetwork 用 MCTS 搜索分布训练（交叉熵），而非 REINFORCE（更稳定）
    3. GaussianParameterNetwork 正确处理连续参数空间
    4. 温度参数 τ 控制探索/利用平衡，支持退火策略
    5. 经验回放缓冲区提高样本效率
    6. 奖励归一化降低策略梯度方差
    7. 可选的真实 simulate rollout 校准搜索方向
    """
    def __init__(self, transaction_pool, policy_network, param_policy_network,
                 value_network, replay_buffer, reward_normalizer,
                 num_simulations=100, exploration_weight=10, max_iterations=5,
                 rollout_ratio=0.05, initial_temperature=1.0, final_temperature=0.1,
                 replay_batch_size=64):
        self.transaction_pool = transaction_pool
        self.num_actions = len(transaction_pool)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.policy_network = policy_network
        self.param_policy_network = param_policy_network
        self.value_network = value_network
        self.replay_buffer = replay_buffer
        self.reward_normalizer = reward_normalizer
        self.tree = {}
        self.max_iterations = max_iterations
        self.rollout_ratio = rollout_ratio
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.replay_batch_size = replay_batch_size

    # -------------------- MCTS 搜索 --------------------

    def search(self, state, partial_sequence=None, temperature=1.0):
        """
        执行 MCTS 搜索：进行 num_simulations 次模拟，
        返回 (选择的动作, MCTS 搜索分布)。

        搜索分布 π_MCTS 用于：
        1. PolicyNetwork 的训练目标
        2. 存入经验回放缓冲区
        """
        # 第一次模拟确保根节点被创建
        self.simulate(state, partial_sequence)
        # 给根节点添加 Dirichlet 噪声（仅一次，增强探索）
        self._add_dirichlet_noise(state)
        # 剩余模拟
        for _ in range(self.num_simulations - 1):
            self.simulate(state, partial_sequence)

        # 获取搜索后的访问次数分布
        mcts_probs = self.get_mcts_distribution(state)

        # 按温度选择动作
        action = self.select_action_by_temperature(state, temperature)

        return action, mcts_probs

    def simulate(self, state, partial_sequence=None):
        """
        标准 MCTS 单次模拟：Selection → Expansion → Evaluation → Backpropagation

        关键设计（与原版本的区别）：
        - 叶节点评估：ValueNetwork 预测 + 少量真实 rollout（替代 evaluate_DRL）
        - 回溯：仅叶节点价值向上传播，无中间步奖励（标准 MCTS 做法）
        - 消除了"策略网络评估自己置信度"的自我强化循环
        """
        path = []
        current_state = state.copy()
        actions_in_path = []

        while True:
            state_tuple = tuple(current_state)

            # 终止状态
            if self.is_terminal(current_state):
                leaf_value = 0.0
                break

            # 叶节点：扩展 + 评估（Bug6 fix）
            if state_tuple not in self.tree:
                action_probs = self.policy_network.predict(current_state)
                self.tree[state_tuple] = {
                    "N": [0] * self.num_actions,
                    "W": [0.0] * self.num_actions,
                    "Q": [0.0] * self.num_actions,
                    "P": action_probs.copy()  # 存储副本，后续 Dirichlet 噪声修改不影响原始值
                }

                # 叶节点评估（P3: 混合 ValueNetwork + 真实 rollout）
                if random.random() < self.rollout_ratio:
                    leaf_value = self._rollout(current_state, partial_sequence, actions_in_path)
                else:
                    leaf_value = self.value_network.predict(current_state)
                break

            # Selection: PUCT 选择
            node = self.tree[state_tuple]
            action = self.select_action(current_state, node["P"])
            if action == -1:
                leaf_value = 0.0
                break

            path.append((node, action))
            actions_in_path.append(action)
            current_state = self.next_state(current_state, action)

        # Backpropagation: 将叶节点价值传播回路径上所有祖先节点
        for node, action in reversed(path):
            node["N"][action] += 1
            node["W"][action] += leaf_value
            node["Q"][action] = node["W"][action] / node["N"][action]

        return leaf_value

    def _rollout(self, state, partial_sequence, actions_in_path):
        """
        真实 rollout (P3)：随机完成剩余序列 → 预测参数 → 调用 simulate 获取真实 MEV。
        用少量真实评估校准 MCTS 搜索，防止 ValueNetwork 训练初期不准时搜索偏离。
        """
        committed = list(partial_sequence) if partial_sequence else []
        current_set = set(committed + list(actions_in_path))
        remaining = [i for i in range(self.num_actions) if i not in current_set]
        random.shuffle(remaining)
        full_sequence = committed + list(actions_in_path) + remaining

        try:
            # 用参数网络预测参数（确定性模式，不引入额外噪声）
            param_sampled, _, _ = self.param_policy_network.predict(
                full_sequence, deterministic=True
            )
            params_order = get_params(transactions)
            param_scaled = [p * domain[name][1] for p, name in zip(param_sampled, params_order)]
            mev = self.evaluate(full_sequence, param_scaled)
            return mev if mev is not None else 0.0
        except Exception as e:
            print(f"[Warning] Rollout failed: {e}")
            return 0.0

    def _add_dirichlet_noise(self, state):
        """给根节点的先验概率添加 Dirichlet 噪声，增强搜索探索性"""
        root_tuple = tuple(state)
        if root_tuple not in self.tree:
            return

        node = self.tree[root_tuple]
        valid_actions = [i for i in range(self.num_actions) if state[i] != 1]
        if not valid_actions:
            return

        epsilon = 0.25
        dirichlet_alpha = 0.3
        noise = np.random.dirichlet([dirichlet_alpha] * len(valid_actions))

        for idx, action in enumerate(valid_actions):
            node["P"][action] = (1 - epsilon) * node["P"][action] + epsilon * noise[idx]

    # -------------------- 动作选择 --------------------

    def select_action(self, state, action_probs):
        """PUCT 动作选择（用于 MCTS 内部模拟 Selection 阶段）"""
        node = self.tree[tuple(state)]
        total_N = sum(node["N"])

        valid_actions = [i for i in range(self.num_actions) if state[i] != 1]
        if not valid_actions:
            return -1

        if total_N == 0:
            # 未访问过的节点，用先验概率 P 采样
            valid_P = np.array([node["P"][i] for i in valid_actions], dtype=np.float64)
            valid_P = np.maximum(valid_P, 1e-8)
            valid_P = valid_P / np.sum(valid_P)
            return np.random.choice(valid_actions, p=valid_P)

        # PUCT: Q(s,a) + c · P(s,a) · √N(s) / (1 + N(s,a))
        ucb_values = [
            node["Q"][i] + self.exploration_weight * action_probs[i]
            * np.sqrt(total_N) / (1 + node["N"][i])
            for i in range(self.num_actions)
        ]

        best_action = valid_actions[np.argmax([ucb_values[i] for i in valid_actions])]
        return best_action

    def select_action_by_temperature(self, state, temperature=1.0):
        """
        温度参数控制的动作选择（Bug1 fix: 屏蔽已执行动作）。
        用于 MCTS 搜索完成后的最终动作选择。

        - τ → 0: 贪心选择（exploitation）
        - τ = 1: 按访问次数比例采样（exploration）
        - τ → ∞: 均匀随机
        """
        node = self.tree.get(tuple(state))
        valid_actions = [i for i in range(self.num_actions) if state[i] != 1]

        if not valid_actions:
            return -1

        if node is None:
            return np.random.choice(valid_actions)

        visits = np.array([node["N"][i] for i in valid_actions], dtype=np.float64)

        if np.sum(visits) == 0:
            return np.random.choice(valid_actions)

        if temperature < 1e-8:
            # 贪心：选访问次数最多的
            best_idx = np.argmax(visits)
            return valid_actions[best_idx]

        # 温度缩放的概率采样
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / (visits_temp.sum() + 1e-8)
        return np.random.choice(valid_actions, p=probs)

    def get_mcts_distribution(self, state):
        """获取 MCTS 搜索后的访问次数分布 π_MCTS（用于 PolicyNetwork 训练目标）"""
        node = self.tree.get(tuple(state))
        probs = np.zeros(self.num_actions)
        valid_actions = [i for i in range(self.num_actions) if state[i] != 1]

        if not valid_actions:
            return probs

        if node is None:
            for i in valid_actions:
                probs[i] = 1.0 / len(valid_actions)
            return probs

        total_visits = sum(node["N"][i] for i in valid_actions)
        if total_visits > 0:
            for i in valid_actions:
                probs[i] = node["N"][i] / total_visits
        else:
            for i in valid_actions:
                probs[i] = 1.0 / len(valid_actions)

        return probs

    # -------------------- 辅助方法 --------------------

    def next_state(self, state, action):
        next_state = state.copy()
        next_state[action] = 1
        return next_state

    def is_terminal(self, state):
        return sum(state) >= len(state)

    def evaluate(self, sequence, sample):
        """调用 Hardhat 模拟器计算真实 MEV"""
        temp_transactions = reorder(transactions, sequence)
        params = get_params(transactions)

        sample_dict = {}
        for p_name, v in zip(params, sample):
            sample_dict[p_name] = v * domain_scales[p_name]

        datum = substitute(temp_transactions, sample_dict, cast_to_int=True)

        try:
            mev = simulate(ctx, datum, port_id, involved_dexes, False, '', 'max')
        except Exception as e:
            print(f"[Warning] simulate failed: {e}")
            mev = None

        return mev if mev is not None else 0.0

    # -------------------- 主运行循环 --------------------

    def run(self, initial_state, batch_size=1):
        """
        主运行循环：
        1. MCTS 搜索构建交易排序
        2. GaussianParameterNetwork 预测参数
        3. simulate 评估真实 MEV
        4. 经验存入 ReplayBuffer
        5. 从 ReplayBuffer 采样更新三个网络
        """
        best_sequence_overall = []
        best_reward_overall = float('-inf')

        for iteration in range(self.max_iterations):
            # 温度退火策略
            if self.max_iterations > 1:
                temperature = self.initial_temperature - \
                    (self.initial_temperature - self.final_temperature) * \
                    iteration / (self.max_iterations - 1)
            else:
                temperature = self.initial_temperature

            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.max_iterations}")
            print(f"best_sequence: {[int(x) for x in best_sequence_overall]}")
            print(f"best_MEV: {best_reward_overall}")
            print(f"temperature: {temperature:.4f}")
            print(f"replay_buffer_size: {len(self.replay_buffer)}")
            print(f"reward_normalizer: mean={self.reward_normalizer.mean:.4f}, "
                  f"std={self.reward_normalizer.std:.4f}, count={self.reward_normalizer.count}")
            print(f"{'='*60}")

            # Bug2 fix: 在 batch 循环外初始化，跨 batch 积累轨迹
            all_trajectories = []

            for b in range(batch_size):
                best_sequence = []
                episode_mcts_data = []  # 收集 (state, mcts_probs) 对
                state = initial_state
                self.tree = {}  # 每个 episode 重置搜索树

                start = time.time()

                # MCTS 搜索：逐步构建交易序列
                while not self.is_terminal(state):
                    action, mcts_probs = self.search(
                        state,
                        partial_sequence=best_sequence,
                        temperature=temperature
                    )
                    if action is None or action == -1:
                        break

                    episode_mcts_data.append((state.copy(), mcts_probs.copy()))
                    best_sequence.append(action)
                    state = self.next_state(state, action)

                result = [int(x) for x in best_sequence]
                print(f"\nSearch sequence: {result}")
                print(f"MCTS搜索用时: {time.time() - start:.4f} 秒")

                # 参数预测（高斯采样）
                start = time.time()
                param_sampled, param_mean, param_log_std = \
                    self.param_policy_network.predict(best_sequence, deterministic=False)
                print(f"参数预测用时: {time.time() - start:.4f} 秒")

                params_order = get_params(transactions)
                param_scaled = [p * domain[name][1]
                                for p, name in zip(param_sampled, params_order)]
                print(f"Sampled params (scaled): "
                      f"{[float(f'{x:.4f}') for x in param_scaled]}")

                # 真实 MEV 评估
                start = time.time()
                reward = self.evaluate(best_sequence, param_scaled)
                print(f"MEV Evaluation用时: {time.time() - start:.4f} 秒")

                # 更新奖励归一化器（仅用新的真实奖励更新统计量）
                self.reward_normalizer.update(reward)

                # 存入经验回放缓冲区
                for (s, mp) in episode_mcts_data:
                    self.replay_buffer.push_policy_value(s, mp, reward)
                self.replay_buffer.push_param(
                    best_sequence, param_sampled.tolist(), reward
                )

                # Bug2 fix: 累积所有 batch 的轨迹
                all_trajectories.append({
                    'sequence': best_sequence,
                    'reward': reward,
                })

                print(f"\n{'='*50}")
                print(f"Iteration: {iteration}, Batch: {b}")
                print(f"Sequence: {result}")
                print(f"MEV Reward: {reward}")
                print(f"{'='*50}")

                if reward > best_reward_overall:
                    best_reward_overall = reward
                    best_sequence_overall = best_sequence[:]

            # ===== 从经验回放缓冲区采样并更新三个网络 =====
            self._update_networks()

        return best_sequence_overall, best_reward_overall

    def _update_networks(self):
        """
        从经验回放缓冲区采样并更新所有网络。
        P4 fix: 每个网络仅更新一次（而非原来的 N 次循环）。
        改进: 增加内部训练轮数 (epochs=10)，充分利用采集到的数据，提高样本效率。
        """
        batch_size = min(self.replay_batch_size, len(self.replay_buffer))
        if batch_size == 0:
            return

        # 增加训练轮数，让网络多学几次
        epochs = 10
        total_policy_loss = 0
        total_value_loss = 0
        total_param_loss = 0

        for _ in range(epochs):
            # 每次重新采样，或者固定这批数据训练多次（这里选择重新采样以增加随机性）
            # 如果 buffer 很小，重新采样其实差别不大；如果 buffer 很大，重新采样能覆盖更多数据。
            # 为了稳定，我们在这个小循环里针对同一批数据进行多次梯度下降也是可以的，
            # 但考虑到 ReplayBuffer 的随机性，每次采样新数据可能更好。
            # 这里采用：每次采样新 batch 进行更新。
            
            pv_batch = self.replay_buffer.sample_policy_value(batch_size)
            param_batch = self.replay_buffer.sample_param(batch_size)

            states_batch = [x[0] for x in pv_batch]
            probs_batch = [x[1] for x in pv_batch]
            mev_batch = [x[2] for x in pv_batch]

            # 1. 更新 PolicyNetwork（交叉熵 with MCTS 分布）
            policy_loss = self.policy_network.update_with_mcts_probs(
                states_batch, probs_batch
            )
            total_policy_loss += policy_loss

            # 2. 更新 ValueNetwork（MSE，目标为真实 MEV）
            value_loss = self.value_network.update_batch(
                states_batch, mev_batch
            )
            total_value_loss += value_loss

            # 3. 更新 GaussianParameterNetwork（REINFORCE）
            param_loss = 0.0
            if param_batch:
                seq_batch = [x[0] for x in param_batch]
                act_batch = [x[1] for x in param_batch]
                rew_batch = [x[2] for x in param_batch]

                # 用 RewardNormalizer 计算 advantage（不更新统计量）
                advantages = [self.reward_normalizer.normalize(r) for r in rew_batch]

                param_loss = self.param_policy_network.update_batch(
                    seq_batch, act_batch, advantages
                )
                total_param_loss += param_loss

        # 打印平均 Loss
        print(f"[Network Update] PolicyLoss: {total_policy_loss/epochs:.4f}, "
              f"ValueLoss: {total_value_loss/epochs:.6f}, ParamLoss: {total_param_loss/epochs:.4f}")


# ============================================================
# 辅助函数
# ============================================================
def reorder(transactions, order):
    '''
        function to reorder a set of transactions, except for the first one
    '''
    reordered_transactions = [transactions[0]] + [transactions[i+1] for i in order]
    return reordered_transactions


# ============================================================
# 主程序
# ============================================================
args = parse_args()

result_path = args.output
folder_path = args.address

dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

for file in dirs:

    transactions_addr = os.path.join(folder_path, file, "amm_reduced")
    domain_addr = os.path.join(folder_path, file, "domain")

    os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_path, f"run_log_{file}.txt")

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print(f"[*] Log started at {timestamp}")
    print(f"Address: {transactions_addr}")
    print(f"Output: {os.path.join(result_path, file)}")

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
        lower_lim, upper_lim = float(tokens[1]), float(tokens[2])
        token_pair = new_domain.split('/')[-2]
        if len(tokens) == 3:
            VALID_RANGE[token_pair] = min(1e6, upper_lim)
            if upper_lim > VALID_RANGE[token_pair]:
                domain_scales[tokens[0]] = upper_lim / VALID_RANGE[token_pair]
                upper_lim = VALID_RANGE[token_pair]
            else:
                domain_scales[tokens[0]] = 1.0
        else:
            assert len(tokens) == 4
            domain_scales[tokens[0]] = float(tokens[3])
        domain[tokens[0]] = (lower_lim, upper_lim)

    print(domain)
    print(domain_scales)

    transactions = transactions_f.readlines()
    print("Setting up...", transactions[0])
    ctx = setup(transactions[0], capital=10000)

    involved_dexes = ['sushiswap', 'aave', 'uniswapv3']

    print("DEXs: ", involved_dexes)

    ctx = prepare_once(ctx, transactions, port_id, involved_dexes)
    print("Prices: ", ctx.prices)

    input_size = len(transactions[1:])

    print("Input size: ", input_size)
    print("")

    # 创建三个网络
    policy_network = PolicyNetwork(
        input_size=input_size, learning_rate=args.lr_policy
    )
    value_network = ValueNetwork(
        input_size=input_size, learning_rate=args.lr_value
    )
    param_policy_network = GaussianParameterNetwork(
        input_size=input_size, param_size=len(domain),
        learning_rate=args.lr_param
    )

    # 创建经验回放缓冲区和奖励归一化器
    replay_buffer = ReplayBuffer(capacity=args.replay_capacity)
    reward_normalizer = RewardNormalizer()

    # 创建 MCTS+DRL 求解器
    solver = MCTS_DRLSolver(
        transactions[1:], policy_network, param_policy_network,
        value_network, replay_buffer, reward_normalizer,
        num_simulations=args.num_simulations,
        exploration_weight=args.exploration_weight,
        max_iterations=args.max_iterations,
        rollout_ratio=args.rollout_ratio,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        replay_batch_size=args.replay_batch_size
    )

    # 定义初始状态 (所有交易尚未执行)
    initial_state = [0] * input_size

    # 运行 MCTS + DRL 优化
    best_sequence, reward = solver.run(initial_state)

    print("Best sequence:", [int(x) for x in best_sequence])
    print("Best Reward:", reward)

    with open(os.path.join(result_path, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"File: {file}\n")
        f.write(f"Best sequence: {[int(x) for x in best_sequence]}\n")
        f.write(f"Best Reward: {reward}\n\n")