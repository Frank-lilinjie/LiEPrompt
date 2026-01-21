'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from timm.models.layers.weight_init import trunc_normal_

# 路由器层
class LinearRouter(nn.Module):
    def __init__(self, embed_dim, depth):
        super(LinearRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.depth = depth
        self.embed_dim = embed_dim
        # 可训练的高斯噪声参数
        self.noise_mean = nn.Parameter(torch.zeros(1))            # 初始化为 0
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)         # 初始化为较小扰动
        self.register_buffer('score', torch.zeros(depth))

    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score.detach().cpu().numpy()
    
    def forward(self, x_embed):
        gate_logits = self.Router(x_embed)
        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise
        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        # gate_avg = gate.mean(dim=0)                  # [num_layers]
        with torch.no_grad():
            batch_score = gate.mean(dim=0)
            self.score.copy_(batch_score)
        return gate


class KlLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth):
        super(KlLinearRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.depth = depth
        # 可训练的高斯噪声参数
        self.noise_mean = nn.Parameter(torch.zeros(1))            
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x_embed, train=False):
        """
        Args:
            x_embed: [B, input_dim] image-level representation
        Returns:
            gate: [B, num_layers], soft importance for each layer
            gate_avg: [num_layers], average importance across batch
            kl_loss: scalar, load-balancing loss encouraging uniformity
        """
        gate_logits = self.Router(x_embed)

        if train:
            noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
            gate_logits = gate_logits + noise

        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        gate_avg = gate.mean(dim=0)                  # [num_layers]

        # 计算 KL 散度，目标是均匀分布
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        kl_loss = F.kl_div(gate_avg.log(), uniform, reduction='batchmean')

        return gate, gate_avg, kl_loss


class UniformSimpleRouter(nn.Module):
    def __init__(self, embed_dim, depth, ema_decay=0.9):
        super(UniformSimpleRouter, self).__init__()
        self.Router = nn.Linear(embed_dim, depth)
        self.depth = depth
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.noise_mean = nn.Parameter(torch.zeros(1))            
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)
        self.task_router_dict = {}
        self.register_buffer('score_ema', torch.zeros(depth))  # [L]
        self.ema_initialized = False

    def save_task_state(self, task_id):
        self.task_router_dict[task_id] = {
            'router_weight': self.Router.state_dict(),
        }
        self.Router = self._build_router(self.noise_std.device)

    def _build_router(self, device):
        router = nn.Linear(self.embed_dim, self.depth).to(device)
        return router
    
    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score_ema.detach().cpu().numpy()
    
    def forward_train(self, x_embed):
        gate_logits = self.Router(x_embed)
        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise

        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        gate_avg = gate.mean(dim=0)                  # [num_layers]

        # 直接用均方误差约束为均匀分布
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        uniform_loss = F.mse_loss(gate_avg, uniform)

        with torch.no_grad():
            batch_score = gate.mean(dim=0)  # [L]
            if not self.ema_initialized:
                self.score_ema.copy_(batch_score)
                self.ema_initialized = True
            else:
                self.score_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_score)

        return gate, uniform_loss

    def forward_test(self, x_embed, task_ids, use_mask=True):
        B = x_embed.size(0)
        device = x_embed.device
        gate_out = torch.zeros(B, self.depth, device=device)

        # 处理各种 task_ids 类型为 List[int]
        if isinstance(task_ids, torch.Tensor):
            task_ids = task_ids.view(-1).tolist()
        elif isinstance(task_ids, list) and isinstance(task_ids[0], (list, torch.Tensor)):
            task_ids = [int(t[0]) if isinstance(t, (list, tuple)) else int(t.item()) 
                        for t in task_ids]
        else:
            task_ids = [int(t) for t in task_ids]

        unique_task_ids = set(task_ids)

        for task_id in unique_task_ids:
            # 找出属于当前任务的样本索引
            indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
            x_sub = x_embed[indices]  # [B_sub, D]

            # # 构造一个新的 MLP Router，并载入该任务的参数
            temp_router = self._build_router(device)
            temp_router.load_state_dict(self.task_router_dict[task_id]['router_weight'])

            with torch.no_grad():
                gate_logits = temp_router(x_sub)      # [B_sub, depth]
                gate = F.softmax(gate_logits, dim=-1) # [B_sub, depth]
                # gate = self.importance_mapping(gate_logits)

                if use_mask:
                    # 每个样本自己的均值作为阈值，生成 0/1 mask
                    mean_val = gate.mean(dim=1, keepdim=True)
                    mask = (gate > mean_val).float()
                    gate_out[indices] = mask
                else:
                    gate_out[indices] = gate

        return gate_out

    def forward(self, x_embed, task_ids=None, train=False):
        if train:
            gate, uniform_loss = self.forward_train(x_embed)
            return gate, uniform_loss
        else:
            gate = self.forward_test(x_embed, task_ids)
            return gate


class UniformLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth, ema_decay=0.9, hidden_dim=256):
        super(UniformLinearRouter, self).__init__()
        self.Router = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # 或者 nn.ReLU()
            nn.Linear(hidden_dim, depth)
        )
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.noise_mean = nn.Parameter(torch.zeros(1))            
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)
        self.task_router_dict = {}
        self.register_buffer('score_ema', torch.zeros(depth))  # [L]
        self.ema_initialized = False

    def save_task_state(self, task_id):
        self.task_router_dict[task_id] = {
            'router_weight': self.Router.state_dict(),
        }
        self.Router = self._build_router(self.noise_std.device)

    def _build_router(self, device):
        """
        构造一个新的两层 MLP Router，用于 init 或 forward_test 中重建。
        """
        router = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.depth)
        ).to(device)
        return router
    
    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score_ema.detach().cpu().numpy()
    
    def forward_train(self, x_embed):
        gate_logits = self.Router(x_embed)
        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise

        gate = F.softmax(gate_logits, dim=-1)        # [B, num_layers]
        # gate = self.importance_mapping(gate_logits)
        gate_avg = gate.mean(dim=0)                  # [num_layers]

        # 直接用均方误差约束为均匀分布
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        uniform_loss = F.mse_loss(gate_avg, uniform)

        with torch.no_grad():
            batch_score = gate.mean(dim=0)  # [L]
            if not self.ema_initialized:
                self.score_ema.copy_(batch_score)
                self.ema_initialized = True
            else:
                self.score_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_score)

        return gate, uniform_loss

    def forward_test(self, x_embed, task_ids, use_mask=True):
        B = x_embed.size(0)
        device = x_embed.device
        gate_out = torch.zeros(B, self.depth, device=device)

        # 处理各种 task_ids 类型为 List[int]
        if isinstance(task_ids, torch.Tensor):
            task_ids = task_ids.view(-1).tolist()
        elif isinstance(task_ids, list) and isinstance(task_ids[0], (list, torch.Tensor)):
            task_ids = [int(t[0]) if isinstance(t, (list, tuple)) else int(t.item()) 
                        for t in task_ids]
        else:
            task_ids = [int(t) for t in task_ids]

        unique_task_ids = set(task_ids)

        for task_id in unique_task_ids:
            # 找出属于当前任务的样本索引
            indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
            x_sub = x_embed[indices]  # [B_sub, D]

            # # 构造一个新的 MLP Router，并载入该任务的参数
            temp_router = self._build_router(device)
            temp_router.load_state_dict(self.task_router_dict[task_id]['router_weight'])

            with torch.no_grad():
                gate_logits = temp_router(x_sub)      # [B_sub, depth]
                gate = F.softmax(gate_logits, dim=-1) # [B_sub, depth]
                # gate = self.importance_mapping(gate_logits)

                if use_mask:
                    # 每个样本自己的均值作为阈值，生成 0/1 mask
                    mean_val = gate.mean(dim=1, keepdim=True)
                    mask = (gate > mean_val).float()
                    gate_out[indices] = mask
                else:
                    gate_out[indices] = gate

        return gate_out

    def forward(self, x_embed, task_ids=None, train=False):
        if train:
            gate, uniform_loss = self.forward_train(x_embed)
            return gate, uniform_loss
        else:
            gate = self.forward_test(x_embed, task_ids)
            return gate

class RandomLinearRouter(nn.Module):
    def __init__(self, embed_dim, depth, selected_layers=[4,5,6]):
        """
        Args:
            embed_dim: 输入特征维度
            depth: 总层数（例如 ViT 有 12 层）
            selected_layers: List[int]，你想激活的层，比如 [2, 5, 10]
        """
        super(RandomLinearRouter, self).__init__()
        self.depth = depth
        self.selected_layers = selected_layers

        # 构造静态 one-hot 层权重（multi-hot）
        gate_vec = torch.zeros(depth)
        gate_vec[selected_layers] = 1.0
        self.register_buffer('gate_template', gate_vec.view(1, -1))  # shape [1, depth]

    def forward(self, x_embed, train=False):
        """
        Args:
            x_embed: [B, input_dim]，但这里我们不使用 x_embed
        Returns:
            gate: [B, depth]，每个样本都使用固定的 gate
            gate_avg: [depth]，等于 gate[0]，因为每个样本都一样
        """
        B = x_embed.size(0)
        gate = self.gate_template.expand(B, -1)  # shape: [B, depth]
        gate_avg = self.gate_template.squeeze(0)
        return gate, gate_avg

class SimpleRouter(nn.Module):
    """
    最简化版路由器：无噪声、无任务参数保存、无均匀分布损失
    用于对比实验验证这些机制的必要性
    """
    def __init__(self, embed_dim, depth, ema_decay=0, hidden_dim=256):
        super(SimpleRouter, self).__init__()
        self.Router = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, depth)
        )
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.register_buffer('score_ema', torch.zeros(depth))
        self.ema_initialized = False

    def get_running_score(self):
        """获取当前的 EMA 平滑层重要性得分（用于可视化）"""
        return self.score_ema.detach().cpu().numpy()

    def forward(self, x_embed):
        """
        简单的前向传播

        Args:
            x_embed: [B, embed_dim]
            task_ids: 忽略，保持接口一致性
            train: 忽略，保持接口一致性

        Returns:
            gate: [B, depth] 层权重
        """
        gate_logits = self.Router(x_embed)
        gate = F.softmax(gate_logits, dim=-1)  # [B, depth]

        # 更新 EMA 统计（仅用于可视化）
        with torch.no_grad():
            batch_score = gate.mean(dim=0)
            if not self.ema_initialized:
                self.score_ema.copy_(batch_score)
                self.ema_initialized = True
            else:
                self.score_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * batch_score)

        return gate


class AttentionCore(nn.Module):
    """
    Attention-based core router:
    输入:  x_embed [B, embed_dim]
    输出:  gate_logits [B, depth]
    """
    def __init__(self, embed_dim, depth, att_dim=None):
        super().__init__()
        if att_dim is None:
            att_dim = embed_dim
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.att_dim = att_dim

        # 将 embedding 映射为 Query
        self.query_proj = nn.Linear(embed_dim, att_dim)

        # 每一层一个 Learnable Layer Token（Key）
        self.layer_tokens = nn.Parameter(
            torch.randn(depth, att_dim) * 0.02
        )

        self.scale = att_dim ** 0.5

    def forward(self, x_embed):
        # [B, att_dim]
        q = self.query_proj(x_embed)

        # [depth, att_dim] → 转置为 [att_dim, depth]
        k = self.layer_tokens.t()

        # scaled dot product attention (logits): [B, depth]
        gate_logits = torch.matmul(q, k) / self.scale
        return gate_logits


class AttentionRouter(nn.Module):
    """
    Attention-based Router 替代 Linear Router：
    - forward_train：训练阶段，返回 gate & uniform_loss
    - forward_test：测试阶段（按 task_id 调取对应 router 参数）
    - save_task_state：保存每个任务的路由参数
    """
    def __init__(self, embed_dim, depth, ema_decay=0.9, att_dim=256):
        super().__init__()

        # 这里换成 AttentionCore
        self.Router = AttentionCore(embed_dim, depth, att_dim=att_dim)

        self.depth = depth
        self.embed_dim = embed_dim
        self.att_dim = self.Router.att_dim
        self.ema_decay = ema_decay

        # 可学习噪声参数
        self.noise_mean = nn.Parameter(torch.zeros(1))
        self.noise_std = nn.Parameter(torch.ones(1) * 0.1)

        # 按 task 保存 router 参数
        self.task_router_dict = {}

        # EMA 平滑重要性
        self.register_buffer('score_ema', torch.zeros(depth))
        self.ema_initialized = False

    def _build_router(self, device):
        """构造新的 AttentionCore 用于新任务重置。"""
        router = AttentionCore(
            embed_dim=self.embed_dim,
            depth=self.depth,
            att_dim=self.att_dim
        ).to(device)
        return router

    def save_task_state(self, task_id):
        """保存当前任务的路由参数，并为下个任务重置 Router。"""
        self.task_router_dict[task_id] = {
            'router_weight': self.Router.state_dict(),
        }
        self.Router = self._build_router(self.noise_std.device)

    def get_running_score(self):
        return self.score_ema.detach().cpu().numpy()
    
    # ---------- Training Forward ----------
    def forward_train(self, x_embed):
        gate_logits = self.Router(x_embed)          # attention logits

        noise = torch.randn_like(gate_logits) * self.noise_std + self.noise_mean
        gate_logits = gate_logits + noise

        gate = F.softmax(gate_logits, dim=-1)       # [B, depth]
        gate_avg = gate.mean(dim=0)

        # 均匀分布约束
        uniform = torch.full_like(gate_avg, 1.0 / self.depth)
        uniform_loss = F.mse_loss(gate_avg, uniform)

        # EMA 更新
        with torch.no_grad():
            batch_score = gate.mean(dim=0)
            if not self.ema_initialized:
                self.score_ema.copy_(batch_score)
                self.ema_initialized = True
            else:
                self.score_ema.mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * batch_score
                )

        return gate, uniform_loss

    # ---------- Testing Forward ----------
    def forward_test(self, x_embed, task_ids, use_mask=True):
        B = x_embed.size(0)
        device = x_embed.device
        gate_out = torch.zeros(B, self.depth, device=device)

        # 整理 task id 为 list[int]
        if isinstance(task_ids, torch.Tensor):
            task_ids = task_ids.view(-1).tolist()
        elif isinstance(task_ids, list) and isinstance(task_ids[0], (list, torch.Tensor)):
            task_ids = [int(t[0]) if isinstance(t, (list, tuple))
                        else int(t.item()) for t in task_ids]
        else:
            task_ids = [int(t) for t in task_ids]

        unique_task_ids = set(task_ids)

        for task_id in unique_task_ids:
            # 找出属于当前 task 的样本索引
            indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
            x_sub = x_embed[indices]

            # 构建并载入该 task 的 router 参数
            temp_router = self._build_router(device)
            temp_router.load_state_dict(self.task_router_dict[task_id]['router_weight'])

            with torch.no_grad():
                gate_logits = temp_router(x_sub)
                gate = F.softmax(gate_logits, dim=-1)

                if use_mask:
                    mean_val = gate.mean(dim=1, keepdim=True)
                    mask = (gate > mean_val).float()    # 0/1 mask
                    gate_out[indices] = mask
                else:
                    gate_out[indices] = gate

        return gate_out

    # ---------- Unified API ----------
    def forward(self, x_embed, task_ids=None, train=False):
        if train:
            return self.forward_train(x_embed)
        else:
            return self.forward_test(x_embed, task_ids)


