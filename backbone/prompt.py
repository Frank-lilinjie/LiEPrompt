import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
import random
from typing import Optional

class MultiPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', 
                 num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, pool_num=None, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.pool_num = pool_num
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.pools = None
        if self.pool_num is not None:
            self.pools = nn.ModuleList()
            for i in range(self.pool_num):
                prompt = PromptModule(self.length, self.embed_dim, self.embedding_key, self.prompt_init, self.prompt_pool, 
                                      self.prompt_key, self.pool_size[i], self.top_k, self.batchwise_prompt, self.prompt_key_init,
                                      self.num_layers, self.use_prefix_tune_for_prompt, self.num_heads, self.same_key_value
                                      )
                self.pools.append(prompt)

    # 传入 x 的 embed 和当前的 layer_id所对应的 prompt_counter 来获得对应的 prompt
    def forward(self, x_embed, prompt_counter=-1, cls_features=None):
        return self.pools[prompt_counter](x_embed, cls_features)
    

class PromptModule(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, same_key_value=False, enlarge_num=0):
        super().__init__()

        self.length = length 
        self.embed_dim = embed_dim 
        self.prompt_pool = prompt_pool 
        self.embedding_key = embedding_key 
        self.prompt_init = prompt_init 
        self.use_prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.prompt_key_init = prompt_key_init
        self.enlarge_num = enlarge_num

        if self.prompt_pool: # 使用提示池，创建一个形状为(pool_size, length, embed_dim)的 prompt
            if self.use_prefix_tune_for_prompt:
                assert self.embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, self.num_heads, self.embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, self.num_heads, self.embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, self.embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if self.use_prompt_key:
            key_shape = (self.pool_size, self.embed_dim) # 拿到这个的
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape)) # 随机的key
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def enlarge_pool_size(self, enlarge_num = None):
        self.enlarge_num = enlarge_num
        new_pool_size = enlarge_num + self.pool_size
        if self.use_prefix_tune_for_prompt:
            assert self.embed_dim % self.num_heads == 0
            if self.same_key_value:
                prompt_pool_shape = (self.num_layers, 1, new_pool_size, self.length, self.num_heads, self.embed_dim // self.num_heads)
                if self.prompt_init == 'zero':
                    prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                prompt = prompt.repeat(1, 2, 1, 1, 1, 1)
            else:
                prompt_pool_shape = (self.num_layers, 2, new_pool_size, self.length, self.num_heads, self.embed_dim // self.num_heads)
                if self.prompt_init == 'zero':
                    prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(prompt, -1, 1)
            prompt.data[:, :, :self.pool_size, :, :, :] = self.prompt.data.clone()

        else:
            prompt_pool_shape = (self.num_layers, new_pool_size, self.length, self.embed_dim)
            if self.prompt_init == 'zero':
                prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif self.prompt_init == 'uniform':
                prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            prompt.data[:, :self.pool_size, :, :] = self.prompt.data.clone()
        
        self.prompt = copy.deepcopy(prompt)
        self.prompt.requires_grad = True  # Enable training for new prompts
        self.prompt.data[:, :, :self.pool_size, :, :, :].requires_grad = False

        # if using learnable prompt keys
        if self.use_prompt_key:
            key_shape = (new_pool_size, self.embed_dim)
            if self.prompt_key_init == 'zero':
                prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif self.prompt_key_init == 'uniform':
                prompt_key = nn.Parameter(torch.randn(key_shape)) # 随机的key
                nn.init.uniform_(prompt_key, -1, 1)
            prompt_key.data[:self.pool_size, :] = self.prompt_key.data.clone()
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(prompt, dim=1)
            prompt_key = prompt_mean
            prompt_key.data[:self.pool_size, :] = self.prompt_key.data.clone()
        
        self.prompt_key = copy.deepcopy(prompt_key)
        self.prompt_key.requires_grad = True 
        self.prompt_key.data[:self.pool_size, :].requires_grad = False

        self.pool_size = new_pool_size


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True) # 计算L2范数
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))) # 使用反平方根进行归一化
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, beta=0):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1) # 平均池化
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] # 最大池化
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1) # 最大池化和平均池化的组合
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=1).to(x_embed.device) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # B, Pool_size

            similarity = similarity.t() # B, pool_size
            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            out['similarity'] = similarity


            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                idx = major_prompt_id.expand(x_embed.shape[0], -1) 
            
            if prompt_mask is not None:
                idx = prompt_mask 

            out['prompt_idx'] = idx

            if self.use_prefix_tune_for_prompt:
                batched_prompt_raw = self.prompt[:,:,idx]  # num_layers, B, top_k, length, C # 这里通过 key 选择到了 prompt
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:,idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx]

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            
            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_prompt:
                assert self.embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, self.embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, self.embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, self.embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt

        return out
    
class Task_MultiPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 task_prompt_key=False, task_pool_size=None, task_topk=None, batchwise_prompt=False, prompt_key_init='uniform', 
                 num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.task_topk = task_topk
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.task_prompt_key = task_prompt_key
        self.prompt_key_init = prompt_key_init
        self.task_pool_size = task_pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.pools = nn.ModuleList()
        self.add_prompt()

    def add_prompt(self):
        prompt = Prompt_Simple(prompt_length=self.length,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                prompt_init=self.prompt_init,
                use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                same_key_value=self.same_key_value,
                prompt_key_init=self.prompt_key_init
                                      )
        self.pools.append(prompt)
        
    
class Feature_MultiPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', 
                num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, same_key_value=False, pool_num = None):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.pool_num = pool_num
        if self.pool_num is not None:
            self.pools = nn.ModuleList()
            for i in range(self.pool_num):
                prompt = PromptModule(self.length, self.embed_dim, self.embedding_key, self.prompt_init, self.prompt_pool, 
                                      self.prompt_key, self.pool_size[i], self.top_k, self.batchwise_prompt, self.prompt_key_init,
                                      self.num_layers, self.use_prefix_tune_for_prompt, self.num_heads, self.same_key_value
                                      )
                self.pools.append(prompt)

    def forward(self, x_embed, prompt_counter=-1, prompt_mask = None, cls_features=None, beta = 0):
        return self.pools[prompt_counter](x_embed, prompt_mask, cls_features, beta)
    

class ADA_Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', 
                prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', 
                num_layers=1, use_prefix_tune_for_prompt=False, num_heads=-1, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.train_layer_idx = [] # 这里存单次训练需要添加的 prompt 的 layer_idx
        self.prompt_layer_idx = [] # 这里存所有添加过 prompt 的 layer_idx
        self.placeholder_idx = []
        self.prompt_train_pools =  nn.ModuleDict()
        self.prompt_all_pools = nn.ModuleDict()# 
        self.place_holder = nn.ModuleDict()

    def update_placeholder(self, layer_idx, num_placeholder):
        for i in layer_idx:
            layer_key = str(i)

            # 如果该层已经存在占位符，则跳过
            if i not in self.placeholder_idx:
                self.placeholder_idx.append(i)

            # 否则新增 num_placeholder 个占位符
            placeholder_list = nn.ModuleList()
            for _ in range(num_placeholder):
                placeholder_list.append(Placeholder())

            # 注册到 self.place_holder 中
            self.place_holder[layer_key] = placeholder_list



    # 更新训练 prompt
    def update_train_prompt_layer(self, layer_idx, old_prompt_ids, add_num):

        layer_key = str(layer_idx)

        if layer_idx not in self.train_layer_idx:
            self.train_layer_idx.append(layer_idx)
        reused_prompt_list = nn.ModuleList()

        # ✅ 尝试从已有 pool 中复用旧 prompt
        if layer_key in self.prompt_all_pools:
            old_prompt_list = self.prompt_all_pools[layer_key]
            assert isinstance(old_prompt_list, nn.ModuleList)
            
            for idx in old_prompt_ids:
                reused_prompt = copy.deepcopy(old_prompt_list[idx])
                reused_prompt_list.append(reused_prompt)

        # ✅ 新增若干 Prompt_Simple（按 add_num）
        for _ in range(add_num):
            new_prompt = Prompt_Simple(
                length=self.length,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                prompt_init=self.prompt_init,
                use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                same_key_value=self.same_key_value,
                use_prompt_key=self.prompt_key,
                prompt_key_init=self.prompt_key_init
            )

            reused_prompt_list.append(new_prompt)

        # ✅ 更新 train pool（List[Prompt_Simple]）
        self.prompt_train_pools[layer_key] = reused_prompt_list

    
    # 更新所有的 prompt
    def update_all_prompt_pools(self, selected_blocks, add_nums):
        """
        将训练结束的 prompt 合并到 prompt_all_pools 中。

        Args:
            selected_blocks (List[int]): 本轮训练的层索引
            add_nums (List[int]): 对应每层新添加的数量
        """
        assert len(selected_blocks) == len(add_nums), "selected_blocks 和 add_nums 长度不一致"
        
        # 添加index 到 self.prompt_layer_idx
        for i in selected_blocks:
            if i not in self.prompt_layer_idx:
                self.prompt_layer_idx.append(i)

        for i, layer_idx in enumerate(selected_blocks):
            layer_key = str(layer_idx)
            new_prompts = self.prompt_train_pools[layer_key]  # List[Prompt_Simple]
            add_num = add_nums[i]

            # 仅保留新添加的尾部 prompt
            new_added_prompts = new_prompts[-add_num:]
            
            if layer_key in self.prompt_all_pools:
                self.prompt_all_pools[layer_key].extend(new_added_prompts)
            else:
                self.prompt_all_pools[layer_key] = new_added_prompts

            # 只冻结本轮新加的 prompt
            for prompt in new_prompts:
                for param in prompt.parameters():
                    param.requires_grad = False
        

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True) # 计算L2范数
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))) # 使用反平方根进行归一化
        return x * x_inv_norm
    
    def forward_train(self, x_embed, layer_idx, cls_features):
        out = dict()

        # 1. 获取样本嵌入
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            x_embed_mean = cls_features if cls_features is not None else torch.max(x_embed, dim=1)[0]
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        # 2. 获取该层的 prompt pool
        prompt_pool = self.prompt_train_pools[layer_idx]
        prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)  # [prompt_num, C]

        # 3. 计算相似度
        prompt_key_norm = self.l2_normalize(prompt_keys, dim=1).to(x_embed.device)  # [P, C]
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)                       # [B, C]
        similarity = torch.matmul(x_embed_norm, prompt_key_norm.T)                  # [B, P]
        out['similarity'] = similarity

        # 4. Top-K prompt 选择
        topk_values, topk_idx = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]

        # 如果 batchwise 共享 prompt：选出全局 top-K
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(topk_idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                fill_n = self.pool_size - prompt_id.shape[0]
                pad_ids = torch.full((fill_n,), torch.min(topk_idx.flatten()), device=topk_idx.device)
                prompt_id = torch.cat([prompt_id, pad_ids])
                id_counts = torch.cat([id_counts, torch.zeros(fill_n, device=topk_idx.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            topk_idx = major_prompt_id.expand(x_embed.shape[0], -1)  # [B, top_k]

        out['prompt_idx'] = topk_idx

        # 5. 构造 prompt batch（无占位逻辑）
        if self.use_prefix_tune_for_prompt:
            B, top_k = topk_idx.shape
            prompt_list = []

            for b in range(B):
                prompt_per_sample = []
                for k in range(top_k):
                    p_idx = topk_idx[b, k].item()
                    prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)
                    prompt_per_sample.append(prompt_tensor.unsqueeze(0))
                # 拼成 [top_k, dual, prompt_len, num_heads, head_dim]
                prompt_per_sample = torch.cat(prompt_per_sample, dim=0)
                prompt_list.append(prompt_per_sample.unsqueeze(0))  # [1, top_k, ...]

            batched_prompt_raw = torch.cat(prompt_list, dim=0)  # [B, top_k, dual, prompt_len, num_heads, head_dim]

            # 变换维度：[dual, B, top_k * prompt_len, num_heads, head_dim]
            B, top_k, num_layers, dual, prompt_len, num_heads, head_dim = batched_prompt_raw.shape
            batched_prompt_raw = batched_prompt_raw.reshape(
                num_layers, B, dual, top_k * prompt_len, num_heads, head_dim
            )  # [dual, B, L, H, D]
        else:
            raise NotImplementedError("Only prefix-tuning is supported currently.")

        # 6. 返回结果
        batched_key_norm = prompt_key_norm[topk_idx]  # [B, top_k, dim]
        batched_prompt = batched_prompt_raw
        out['selected_key'] = batched_key_norm
        out['prompt_key_norm'] = prompt_key_norm
        out['x_embed_norm'] = x_embed_norm
        out['batched_prompt'] = batched_prompt

        return out

    def forward_test(self, x_embed, layer_idx, cls_features):
        out = dict()

        # 1. 获取样本嵌入
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        # 2. 获取该层的 prompt pool 和 placeholder
        prompt_pool = self.prompt_all_pools[layer_idx]

        placeholder_pool = self.place_holder[layer_idx]

        prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)           # [prompt_num, C]
        prompt_keys_length = len(prompt_keys)
        placeholder_keys = torch.cat([p.placeholder for p in placeholder_pool], dim=0) # [placeholder_num, C]

        # 拼接 key：prompt key + 占位符 key
        all_keys = torch.cat([prompt_keys, placeholder_keys], dim=0)                  # [total_num, C]

        # 3. 计算相似度
        prompt_key_norm = self.l2_normalize(all_keys, dim=1).to(x_embed.device)  # [total, C]
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)                    # [B, C]
        similarity = torch.matmul(x_embed_norm, prompt_key_norm.T)               # [B, total]
        out['similarity'] = similarity

        # 4. 判断是否跳过 prompt（如果 top-1 是 placeholder）
        top1_value, top1_idx = torch.topk(similarity, k=1, dim=1)                # [B, 1]
        need_prompt_mask = top1_idx.squeeze(1) < prompt_keys_length              # [B]，True 表示需要添加 prompt

        # 为所有样本计算 Top-K，只从 prompt_key 中取（排除 placeholder）
        prompt_similarity = similarity[:, :prompt_keys_length]                  # [B, prompt_num]
        topk_values, topk_idx = torch.topk(prompt_similarity, k=self.top_k, dim=1)  # [B, top_k]

        # 如果 batchwise prompt：选出全局 top-K
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(topk_idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(topk_idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            topk_idx = major_prompt_id.expand(x_embed.shape[0], -1)  # [B, top_k]

        out['prompt_idx'] = topk_idx

        # 5. 仅对需要添加 prompt 的样本组装 prompt
        if self.use_prefix_tune_for_prompt:
            B, top_k = topk_idx.shape
            prompt_list = []
            for b in range(B):
                if need_prompt_mask[b]:
                    prompt_per_sample = []
                    for k in range(top_k):
                        p_idx = topk_idx[b, k].item()
                        prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)
                        prompt_per_sample.append(prompt_tensor.unsqueeze(0))
                    prompt_per_sample = torch.cat(prompt_per_sample, dim=0)  # [top_k, dual, prompt_len, num_heads, head_dim]
                    prompt_list.append(prompt_per_sample.unsqueeze(0))       # [1, top_k, ...]
                else:
                    # 若不需要 prompt，则占位（全 0），稍后外部 forward 判断是否启用
                    dummy = torch.zeros((1, top_k, *prompt_pool[0].prompt.shape), device=x_embed.device)
                    prompt_list.append(dummy)

            batched_prompt_raw = torch.cat(prompt_list, dim=0)  # [B, top_k, dual, prompt_len, num_heads, head_dim]

            # 转换维度：[num_layers, B, dual, top_k * prompt_len, num_heads, head_dim]
            B, top_k, num_layers, dual, prompt_len, num_heads, head_dim = batched_prompt_raw.shape
            # batched_prompt_raw = batched_prompt_raw.permute(2, 0, 1, 3, 4, 5)  # [dual, B, top_k, prompt_len, num_heads, head_dim]
            batched_prompt_raw = batched_prompt_raw.reshape(
                num_layers, B, dual, top_k * prompt_len, num_heads, head_dim
            )

        batched_prompt = batched_prompt_raw
        batched_key_norm = prompt_key_norm[topk_idx]  # [B, top_k, dim]

        out['selected_key'] = batched_key_norm
        out['prompt_key_norm'] = prompt_key_norm
        out['x_embed_norm'] = x_embed_norm
        out['batched_prompt'] = batched_prompt
        out['need_prompt_mask'] = need_prompt_mask  # 加上这个字段，方便外部使用

        return out
    
    def forward(self, x_embed, layer_idx, test=False, cls_features=None):
        if test:
            out = self.forward_test(x_embed, layer_idx, cls_features)
        else:
            out = self.forward_train(x_embed, layer_idx, cls_features)
        return out


class Prompt_Simple(nn.Module):
    def __init__(self, length = 5, embed_dim = 768, 
                 num_heads = -1, prompt_init = 'uniform',
                 use_prefix_tune_for_prompt = False, same_key_value = False, 
                 use_prompt_key = True, prompt_key_init = 'uniform'):
        """
        Simple prompt module that holds prompt and optional key.

        Args:
            num_prompts: Number of prompts.
            prompt_length: Length of each prompt (token-wise).
            embed_dim: Embedding dimension of each token.
            use_key: Whether to initialize key for each prompt.
        """
        super().__init__()
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        # 初始化 prompt
        if use_prefix_tune_for_prompt:
            assert embed_dim % num_heads == 0
            # key 和 value 使用同一份参数
            if same_key_value:
                prompt_pool_shape = (1, 1, length, num_heads, embed_dim // num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
            else:
                prompt_pool_shape = (1, 2, length, num_heads, embed_dim // num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
        else:
            prompt_pool_shape = (1, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # 初始化key
        if use_prompt_key:
            key_shape = (1, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape)) # 随机的key
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def forward_features(self):
        # Flatten 所有 prompt token
        if self.use_prefix_tune_for_prompt:
            return self.prompt[0,0,:,:,:]
        else:
            return self.prompt.mean(dim=0)
        
class Placeholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.randn(1, 768))


class MoEprompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform',
            prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
            num_layers=1, use_prefix_tune_for_prompt=False,use_prompt_key=False, num_heads=-1, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.use_prompt_key = use_prompt_key
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.prompt_pools =  nn.ModuleDict()
        self.MoeRouter = nn.ModuleList()
        # 记录每层旧 prompt 的数量（与 MixPrompt 对齐）
        self.old_prompt_counts = {}

    def init_train_prompt_pool(self, layer_pool_size=5):
        for i in range(self.num_layers):
            layer_prompt_pool = nn.ModuleList()
            router = nn.Linear(self.embed_dim, layer_pool_size)
            self.MoeRouter.append(router)

            for j in range(layer_pool_size):
                layer_prompt_pool.append(Prompt_Simple(length=self.length, embed_dim=self.embed_dim,
                                                num_heads=self.num_heads, prompt_init=self.prompt_init,
                                                use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                                                same_key_value=self.same_key_value,
                                                use_prompt_key=self.use_prompt_key,
                                                prompt_key_init=self.prompt_key_init))
            self.prompt_pools[str(i)] = layer_prompt_pool
            # 初始化时所有 prompt 都是"旧"的
            self.old_prompt_counts[str(i)] = layer_pool_size

    def update_prompt_layer(self, layer_idx: list, add_num=1):
        """
        扩展指定层的 prompt pool（与 MixPrompt API 对齐）

        Args:
            layer_idx: 需要扩展的层索引列表
            add_num: 每层添加的 prompt 数量
        """
        for idx in layer_idx:
            layer_key = str(idx)
            pool = self.prompt_pools[layer_key]
            device = next(pool[0].parameters()).device if len(pool) > 0 else torch.device("cpu")

            # 记录扩展前的 prompt 数量
            old_count = len(pool)

            # 冻结旧 prompt
            for prompt in pool:
                for param in prompt.parameters():
                    param.requires_grad = False

            # 添加新 prompt
            for _ in range(add_num):
                new_prompt = Prompt_Simple(
                    length=self.length,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    prompt_init=self.prompt_init,
                    use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                    same_key_value=self.same_key_value,
                    use_prompt_key=self.use_prompt_key,
                    prompt_key_init=self.prompt_key_init
                ).to(device)
                pool.append(new_prompt)

            # 扩展 Router 输出维度
            old_router = self.MoeRouter[idx]
            new_pool_size = len(pool)
            new_router = nn.Linear(self.embed_dim, new_pool_size).to(device)

            # 复制旧权重
            with torch.no_grad():
                new_router.weight.data[:old_count] = old_router.weight.data
                new_router.bias.data[:old_count] = old_router.bias.data
                # 新增的输出用小随机值初始化
                nn.init.normal_(new_router.weight.data[old_count:], std=0.01)
                nn.init.zeros_(new_router.bias.data[old_count:])

            self.MoeRouter[idx] = new_router

            # 更新旧 prompt 数量记录
            self.old_prompt_counts[layer_key] = old_count

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """L2 归一化（与 MixPrompt 对齐）"""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, layer_idx, cls_features = None):
        out = dict()
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            x_embed_mean = cls_features if cls_features is not None else torch.max(x_embed, dim=1)[0]
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        prompt_pool = self.prompt_pools[layer_idx]
        router = self.MoeRouter[int(layer_idx)]

        gate_logits = router(x_embed_mean)
        gate_scores = F.softmax(gate_logits, dim=-1)
        out['router_score'] = gate_scores

        # 计算正交损失（与 MixPrompt 对齐）
        old_count = self.old_prompt_counts.get(layer_idx, len(prompt_pool))
        new_old_orthogonal_loss = torch.tensor(0.0, device=x_embed.device)

        if old_count < len(prompt_pool) and self.use_prompt_key:
            # 存在新旧 prompt 之分
            prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)
            prompt_key_norm = self.l2_normalize(prompt_keys, dim=1)

            old_keys = prompt_key_norm[:old_count]
            new_keys = prompt_key_norm[old_count:]

            # 新旧 prompt 正交损失
            new_old_sim = torch.matmul(new_keys, old_keys.T)
            new_old_orthogonal_loss = torch.mean(new_old_sim ** 2)

            # 新 prompt 之间的多样性损失
            if len(new_keys) > 1:
                new_new_gram = torch.matmul(new_keys, new_keys.T)
                new_identity = torch.eye(len(new_keys), device=new_new_gram.device)
                new_diversity_loss = F.mse_loss(new_new_gram, new_identity)
                new_old_orthogonal_loss = new_old_orthogonal_loss + new_diversity_loss

        out['new_old_orthogonal_loss'] = new_old_orthogonal_loss

        # 全局正交损失
        if self.use_prompt_key:
            prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)
            prompt_key_norm = self.l2_normalize(prompt_keys, dim=1)
            P = prompt_key_norm.size(0)
            gram_matrix = torch.matmul(prompt_key_norm, prompt_key_norm.T)
            identity = torch.eye(P, device=gram_matrix.device)
            orthogonality_loss = F.mse_loss(gram_matrix, identity)
            out['separation_loss'] = orthogonality_loss
        else:
            out['separation_loss'] = torch.tensor(0.0, device=x_embed.device)

        # MoE 加权融合 prompt
        prompt_raw = torch.cat([p.prompt.to(x_embed.device) for p in prompt_pool], dim=0)
        prompt_exp = prompt_raw.unsqueeze(0).expand(gate_scores.size(0), -1, -1, -1, -1, -1)
        gate_scores_exp = gate_scores.view(gate_scores.size(0), -1, 1, 1, 1, 1)
        prompt_weighted = gate_scores_exp * prompt_exp
        batched_prompt = prompt_weighted.sum(dim=1)
        out['batched_prompt'] = batched_prompt

        # 添加与 MixPrompt 兼容的额外输出
        out['prompt_key_norm'] = None  # MoE 不使用 key-based 选择
        out['x_embed_norm'] = self.l2_normalize(x_embed_mean, dim=1) if self.use_prompt_key else None
        out['similarity'] = None  # MoE 使用 router，没有 similarity

        return out     

class MixPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform',
            prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
            num_layers=1, use_prefix_tune_for_prompt=False,use_prompt_key=False, num_heads=-1, same_key_value=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.use_prefix_tune_for_prompt = use_prefix_tune_for_prompt
        self.use_prompt_key = use_prompt_key
        self.prompt_init = prompt_init
        self.same_key_value = same_key_value
        self.prompt_pools =  nn.ModuleDict()
        # 记录每层旧 prompt 的数量（用于计算新旧 prompt 正交损失）
        self.old_prompt_counts = {}
    
    def init_train_prompt_pool(self, layer_pool_size=5):
        for i in range(self.num_layers):
            layer_prompt_pool = nn.ModuleList()
            for j in range(layer_pool_size):
                layer_prompt_pool.append(Prompt_Simple(length=self.length, embed_dim=self.embed_dim,
                                                num_heads=self.num_heads, prompt_init=self.prompt_init,
                                                use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                                                same_key_value=self.same_key_value,
                                                use_prompt_key=self.use_prompt_key,
                                                prompt_key_init=self.prompt_key_init))
            self.prompt_pools[str(i)] = layer_prompt_pool
            # 初始化时所有 prompt 都是"旧"的
            self.old_prompt_counts[str(i)] = layer_pool_size
    
    def update_prompt_layer(self, layer_idx: list, add_num=1):
        for idx in layer_idx:
            layer_key = str(idx)
            pool = self.prompt_pools[layer_key]
            device = next(pool[0].parameters()).device if len(pool) > 0 else torch.device("cpu")

            # 记录扩展前的 prompt 数量
            old_count = len(pool)

            for prompt in pool:
                for param in prompt.parameters():
                    param.requires_grad = False
            # ✅ 新增若干 Prompt_Simple（按 add_num）
            for _ in range(add_num):
                new_prompt = Prompt_Simple(
                    length=self.length,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    prompt_init=self.prompt_init,
                    use_prefix_tune_for_prompt=self.use_prefix_tune_for_prompt,
                    same_key_value=self.same_key_value,
                    use_prompt_key=self.use_prompt_key,
                    prompt_key_init=self.prompt_key_init
                ).to(device)

                pool.append(new_prompt)

            # 更新旧 prompt 数量记录
            self.old_prompt_counts[layer_key] = old_count
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True) # 计算L2范数
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))) # 使用反平方根进行归一化
        return x * x_inv_norm
    
    def forward(self, x_embed, layer_idx, cls_features = None):
        out = dict()
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            x_embed_mean = cls_features if cls_features is not None else torch.max(x_embed, dim=1)[0]
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        
        prompt_pool = self.prompt_pools[layer_idx]
        prompt_keys = torch.cat([p.prompt_key for p in prompt_pool], dim=0)  # [prompt_num, C]

        # 3. 计算相似度
        prompt_key_norm = self.l2_normalize(prompt_keys, dim=1).to(x_embed.device)  # [P, C]
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)                       # [B, C]
        similarity = torch.matmul(x_embed_norm, prompt_key_norm.T)                  # [B, P]
        out['similarity'] = similarity

        # 计算正交损失：新 prompt 应该与旧 prompt 正交
        old_count = self.old_prompt_counts.get(layer_idx, len(prompt_pool))
        new_old_orthogonal_loss = torch.tensor(0.0, device=x_embed.device)

        if old_count < len(prompt_pool):
            # 存在新旧 prompt 之分
            old_keys = prompt_key_norm[:old_count]  # [old_P, C]
            new_keys = prompt_key_norm[old_count:]  # [new_P, C]

            # 计算新旧 prompt key 之间的余弦相似度矩阵
            # 目标：新旧 prompt 之间应该尽可能正交（相似度接近0）
            new_old_sim = torch.matmul(new_keys, old_keys.T)  # [new_P, old_P]

            # 正交损失：希望新旧 prompt 之间的相似度接近 0
            new_old_orthogonal_loss = torch.mean(new_old_sim ** 2)

            # 新 prompt 之间也应该保持多样性（互相正交）
            if len(new_keys) > 1:
                new_new_gram = torch.matmul(new_keys, new_keys.T)  # [new_P, new_P]
                new_identity = torch.eye(len(new_keys), device=new_new_gram.device)
                new_diversity_loss = F.mse_loss(new_new_gram, new_identity)
                new_old_orthogonal_loss = new_old_orthogonal_loss + new_diversity_loss

        out['new_old_orthogonal_loss'] = new_old_orthogonal_loss

        # 全局正交损失（所有 prompt 之间）
        P = prompt_key_norm.size(0)
        gram_matrix = torch.matmul(prompt_key_norm, prompt_key_norm.T)  # [P, P]
        identity = torch.eye(P, device=gram_matrix.device)
        orthogonality_loss = F.mse_loss(gram_matrix, identity)
        out['separation_loss'] = orthogonality_loss

        # 根据 top_k 值选择不同的 prompt 融合方式
        if self.top_k > 0:
            # top_k > 0: 选择 top-k 个 prompt 进行 concat 拼接
            topk_values, topk_idx = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]

            # 如果 batchwise 共享 prompt：选出全局 top-K
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(topk_idx, return_counts=True, sorted=True)
                if prompt_id.shape[0] < len(prompt_pool):
                    fill_n = len(prompt_pool) - prompt_id.shape[0]
                    pad_ids = torch.full((fill_n,), torch.min(topk_idx.flatten()), device=topk_idx.device)
                    prompt_id = torch.cat([prompt_id, pad_ids])
                    id_counts = torch.cat([id_counts, torch.zeros(fill_n, device=topk_idx.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)
                major_prompt_id = prompt_id[major_idx]
                topk_idx = major_prompt_id.expand(x_embed.shape[0], -1)  # [B, top_k]

            out['prompt_idx'] = topk_idx

            # 构造 prompt batch - concat 方式
            if self.use_prefix_tune_for_prompt:
                B, top_k = topk_idx.shape
                prompt_list = []

                for b in range(B):
                    prompt_per_sample = []
                    for k in range(top_k):
                        p_idx = topk_idx[b, k].item()
                        prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)  # [1, dual, L, H, D]
                        prompt_per_sample.append(prompt_tensor)

                    # concat: [top_k, 1, dual, L, H, D] -> [1, dual, top_k*L, H, D]
                    prompt_per_sample = torch.cat(prompt_per_sample, dim=2)  # 在 L 维度拼接
                    prompt_list.append(prompt_per_sample)

                batched_prompt_raw = torch.cat(prompt_list, dim=0)  # [B, dual, top_k*L, H, D]
            else:
                B, top_k = topk_idx.shape
                prompt_list = []

                for b in range(B):
                    prompt_per_sample = []
                    for k in range(top_k):
                        p_idx = topk_idx[b, k].item()
                        prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)  # [1, L, D]
                        prompt_per_sample.append(prompt_tensor)

                    # concat: [top_k, 1, L, D] -> [1, top_k*L, D]
                    prompt_per_sample = torch.cat(prompt_per_sample, dim=1)  # 在 L 维度拼接
                    prompt_list.append(prompt_per_sample)

                batched_prompt_raw = torch.cat(prompt_list, dim=0)  # [B, top_k*L, D]

        else:
            # top_k == 0: 使用加权混合（weighted mix）方式
            if self.use_prefix_tune_for_prompt:
                B, P = similarity.shape
                prompt_weighted_list = []

                for b in range(B):
                    weights = F.softmax(similarity[b], dim=0)
                    prompts = []

                    for p_idx in range(P):
                        prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)  # [1, dual, L, H, D]
                        prompts.append(prompt_tensor)

                    prompts = torch.cat(prompts, dim=0)  # [P, 1, dual, L, H, D]
                    weights = weights.view(-1, 1, 1, 1, 1)  # [P,1,1,1,1] 用于广播
                    fused_prompt = torch.sum(prompts * weights, dim=0)  # [1, dual, L, H, D]

                    prompt_weighted_list.append(fused_prompt.unsqueeze(0))

                batched_prompt_raw = torch.cat(prompt_weighted_list, dim=0)  # [B, dual, L, H, D]
            else:
                B, P = similarity.shape
                prompt_weighted_list = []

                for b in range(B):
                    weights = F.softmax(similarity[b], dim=0)
                    prompts = []

                    for p_idx in range(P):
                        prompt_tensor = prompt_pool[p_idx].prompt.to(x_embed.device)  # [1, L, D]
                        prompts.append(prompt_tensor)

                    prompts = torch.cat(prompts, dim=0)  # [P, 1, L, D]
                    weights = weights.view(-1, 1, 1)  # [P,1,1]
                    fused_prompt = torch.sum(prompts * weights, dim=0)  # [1, L, D]

                    prompt_weighted_list.append(fused_prompt)

                batched_prompt_raw = torch.cat(prompt_weighted_list, dim=0)  # [B, L, D]

        out['prompt_key_norm'] = prompt_key_norm
        out['x_embed_norm'] = x_embed_norm
        out['batched_prompt'] = batched_prompt_raw

        return out
    

class PromptPoolRouter(nn.Module):
    def __init__(self, 
                 prompt_pool_size,       # N: prompt 数量
                 prompt_len,             # L: 每个 prompt 的长度
                 embed_dim,              # D: embedding 维度
                 num_heads=4,            # 多头路由
                 use_self_attn=True,     # prompt 内部自注意力
                 use_task_token=True):   # 是否使用任务 token 路由
        super().__init__()
        self.N = prompt_pool_size
        self.L = prompt_len
        self.D = embed_dim
        self.num_heads = num_heads
        self.use_self_attn = use_self_attn
        self.use_task_token = use_task_token

        # 初始化 Prompt Pool： [N, L, D]
        self.prompt_pool = nn.Parameter(torch.randn(self.N, self.L, self.D))

        # 多头注意力层：用于 CLS/task token → prompt 的选择
        self.attn_router = nn.MultiheadAttention(embed_dim=self.D, num_heads=self.num_heads, batch_first=True)

        # prompt 内部自注意力（可选）
        if self.use_self_attn:
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.D, nhead=4, batch_first=True),
                num_layers=1
            )

    def forward(self, cls_token: torch.Tensor, task_token: Optional[torch.Tensor] = None):
        """
        cls_token: [B, D]
        task_token: [B, D] or None
        """
        B = cls_token.size(0)
        P = self.prompt_pool.shape[0]

        # prompt_pool_flat: [P, L, D] → [P*L, D]
        prompts = self.prompt_pool  # [P, L, D]

        # step1: 可选的 Prompt 内部自注意力（提升表达）
        if self.use_self_attn:
            prompts = self.self_attn_layer(prompts)  # [P, L, D]

        # step2: 使用 CLS token / Task token 做为 query，路由出融合 prompt
        if self.use_task_token and task_token is not None:
            query = task_token.unsqueeze(1)  # [B, 1, D]
        else:
            query = cls_token.unsqueeze(1)  # [B, 1, D]

        # prompt_key = [P, D]（平均池化），value 仍然是 prompts 本身
        prompt_key = prompts.mean(dim=1)  # [P, D]
        prompt_value = prompts  # [P, L, D]

        # 让所有 prompt 共享同一个 attention 头，重复 batch
        prompt_key = prompt_key.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
        prompt_value = prompt_value.unsqueeze(0).expand(B, -1, -1, -1)  # [B, P, L, D]

        # reshape key/value 为 attention 所需维度
        prompt_key_flat = prompt_key  # [B, P, D]
        prompt_value_flat = prompt_value.reshape(B * P, self.L, self.D)  # [B*P, L, D]

        # attention: query=[B, 1, D], key=[B, P, D], value=[B*P, L, D]
        attn_output, attn_weights = self.attn_router(
            query=query,                # [B, 1, D]
            key=prompt_key,             # [B, P, D]
            value=prompt_key,           # attention 不作用在 value，value 只是给权重用
            need_weights=True
        )  # attn_weights: [B, 1, P]

        attn_weights = attn_weights.squeeze(1)  # [B, P]
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1]

        # weighted combine prompts: [B, P, L, D] × [B, P, 1, 1] → sum over P → [B, L, D]
        fused_prompt = (prompt_value * attn_weights).sum(dim=1)  # [B, L, D]

        return fused_prompt  # 可以直接拼接到每一层的输入上
