import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import LiEpromptNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from collections import defaultdict
import random
import math
import matplotlib.pyplot as plt
from utils.visualize import visualize_layer_importance
from torch.distributions.multivariate_normal import MultivariateNormal

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = LiEpromptNet(args, True)
        self.dataset_name = args["dataset"]
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.pool_size = args["pool_size"]
        self.layer_topk = args["layer_topk"] # 在多少层上添加 prompt
        self.seed = args["seed"]
        self.add_num = args["add_num"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.inc = args["increment"]
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        
        # 记录所有任务的层重要性得分
        self.task_layer_scores = []
        
        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')
        
        # 初始化训练 prompt 池
        self._network.backbone.mixprompt.init_train_prompt_pool(self.pool_size)

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))
    
    def replace_indicator(self, train_loader):
        model = self._network.backbone.eval()
        embedding_list, label_list = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_original(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        class_list = np.unique(train_loader.dataset.labels)

        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.task_indicator.weight.data[class_index] = proto

        
    def after_task(self):
        self._known_classes = self._total_classes
        # 保存当前任务的层重要性得分
        current_layer_score = self._network.backbone.RouterLinear.get_running_score()
        self.task_layer_scores.append(current_layer_score.copy())
        logging.info(f"Task {self._cur_task} layer scores saved: {current_layer_score}")

    def incremental_train(self, data_manager):
        self._network.to(self._device)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._network.update_task_indicator(self._total_classes)

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        # 更新任务标识器
        self.replace_indicator(self.train_loader_for_protonet)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        # 可视化每个任务结束后的层重要性
        avg_layer_score = self._network.compute_layer_importance()

        # 获取 Top-k 层的索引，按重要性从高到低排序
        layer_idx = list(np.argsort(avg_layer_score)[-self.layer_topk:][::-1])  # 从大到小排序取前k个

        logging.info(f"[Prompt扩展] 当前任务完成后，选择第 {layer_idx} 层（Top-{self.layer_topk}重要性）添加 {self.add_num} 个 prompt")
        self._network.backbone.mixprompt.update_prompt_layer(layer_idx=layer_idx, add_num=self.add_num)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


        
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, test_loader, optimizer, scheduler)
        # 保存任务状态
        self._network.backbone.RouterLinear.save_task_state(self._cur_task)
        self._compute_mean(self._network)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)
        
    def get_optimizer(self):
        trainable_params = []
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name} | shape: {tuple(param.shape)}")
                trainable_params.append(param)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                trainable_params, 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                trainable_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        epochs = self.args['tuned_epoch']
        prog_bar = tqdm(range(epochs))

        # 记录每个 epoch 的层得分
        epoch_layer_scores = []

        for _, epoch in enumerate(prog_bar):
            self._network.to(self._device)
            self._network.backbone.train()
            losses = 0.0
            losses_clf = 0.0
            losses_mse = 0.0
            losses_ortho = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs, train=True)
                # logits = output["logits"]
                mse_loss = output["mse_loss"]
                orthogonal_loss = output.get("orthogonal_loss", torch.tensor(0.0, device=self._device))

                # loss_clf = F.cross_entropy(logits, targets)
                
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss_clf = F.cross_entropy(logits, targets.long())

                # 添加正交损失项
                # 权重可以在配置文件中调整，这里默认使用 0.1
                ortho_weight = self.args.get("orthogonal_loss_weight", 0.1)
                mse_weight = self.args.get("mse_loss_weight", 0.1)
                loss = loss_clf +  mse_weight*mse_loss + ortho_weight * orthogonal_loss
                # loss = loss_clf  + ortho_weight * orthogonal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_mse += mse_loss.item()
                losses_ortho += orthogonal_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # 记录当前 epoch 结束时的层得分
            current_layer_score = self._network.backbone.RouterLinear.get_running_score()
            epoch_layer_scores.append(current_layer_score.copy())

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, losses_mse {:.3f}, losses_ortho {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                losses_clf / len(train_loader),
                losses_mse/ len(train_loader),
                losses_ortho / len(train_loader),
                train_acc,
            )
            # info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f},  losses_ortho {:.3f}, Train_accy {:.2f}".format(
            #     self._cur_task,
            #     epoch + 1,
            #     epochs,
            #     losses / len(train_loader),
            #     losses_clf / len(train_loader),
            #     losses_ortho / len(train_loader),
            #     train_acc,
            # )
            prog_bar.set_description(info)

        logging.info(info)

        # 训练结束后绘制层得分变化曲线
        # self.plot_epoch_layer_scores(epoch_layer_scores)

    def _eval_cnn(self, loader):
        self._network.eval()
        
        y_pred_linear, y_pred_proto, y_pred_confidence = [], [], []
        output_pred = []
        y_true = []

        proto_correct, linear_correct, confidence_correct = 0, 0, 0
        total = 0

        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                outputs, proto_logits = self._network(inputs, train=False)
                linear_logits = outputs["logits"][:, :self._total_classes]
                proto_logits = proto_logits[:, :self._total_classes]
                # === 计算置信度（top1 - top2）===
                def compute_confidence(logits):
                    sorted_vals, _ = torch.sort(logits, dim=1, descending=True)
                    return sorted_vals[:, 0] - sorted_vals[:, 1]  # [B]
                conf_linear = compute_confidence(linear_logits)
                conf_proto = compute_confidence(proto_logits)
                use_linear = conf_linear > conf_proto  # [B], bool
                logits_confidence = torch.where(
                    use_linear.unsqueeze(1),  # 扩展为 [B, 1] 用于广播
                    linear_logits,
                    proto_logits
                )
            predicts_linear = torch.topk(linear_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
            predicts_proto = torch.topk(proto_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
            predicts_confidence = torch.topk(logits_confidence, k=self.topk, dim=1, largest=True, sorted=True)[1]
            # === 准确率统计（只看 top-1）===
            linear_correct += (predicts_linear[:, 0].cpu() == targets.cpu()).sum().item()
            proto_correct += (predicts_proto[:, 0].cpu() == targets.cpu()).sum().item()
            confidence_correct += (predicts_confidence[:, 0].cpu() == targets.cpu()).sum().item()
            total += len(targets)

            # === 收集预测结果 ===
            y_pred_linear.append(predicts_linear.cpu().numpy())
            y_pred_proto.append(predicts_proto.cpu().numpy())
            y_pred_confidence.append(predicts_confidence.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        # === 计算准确率并打印 ===
        linear_acc = 100.0 * linear_correct / total
        proto_acc = 100.0 * proto_correct / total
        confidence_acc = 100.0 * confidence_correct / total

        logging.info(f"Linear Classifier Acc: {linear_acc:.2f}%")
        logging.info(f"Prototype Classifier Acc: {proto_acc:.2f}%")
        logging.info(f"Confidence-based Selection Acc: {confidence_acc:.2f}%")

        best_acc = max(linear_acc, proto_acc, confidence_acc)
        if best_acc == linear_acc:
            output_pred = y_pred_linear
        elif best_acc == proto_acc:
            output_pred = y_pred_proto
        else:
            output_pred = y_pred_confidence

        return np.concatenate(output_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, train=False)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def plot_layer_scores(self, gate_history):
        """
        绘制 ViT 12 层在训练过程中每个 batch 的 Router 得分变化趋势，并在线上标注每层的编号

        Args:
            gate_history (List[Tensor or ndarray]): 每个元素为 shape [1, 12] 或 [12]，表示每层的得分
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        # 转换并 squeeze 所有元素为 shape [12]
        processed = []
        for g in gate_history:
            if isinstance(g, torch.Tensor):
                g = g.detach().cpu().numpy()
            g = np.squeeze(g)  # 可能是 [1, 12] -> [12]
            processed.append(g)

        gate_array = np.stack(processed)  # shape: [num_batches, 12]
        num_batches, num_layers = gate_array.shape
        x = np.arange(num_batches)

        plt.figure(figsize=(12, 6))
        colors = plt.cm.get_cmap("tab10", num_layers)

        for layer_id in range(num_layers):
            y = gate_array[:, layer_id]
            plt.plot(x, y, label=f'Layer {layer_id}', color=colors(layer_id))
            
            # 在线注释 —— 只注释最后一个点（靠右），稍微偏移一下
            plt.text(x[-1] + 0.5, y[-1], f"L{layer_id}", 
                    fontsize=8, color=colors(layer_id), 
                    verticalalignment='center')

        plt.xlabel('Batch Index')
        plt.ylabel('Layer Score')
        # plt.title('Layer-wise Score Trend Across Batches')
        plt.grid(True)

        # 可以保留 legend 也可以注释掉
        # plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.dataset_name}_{self.seed}_{self._cur_task}_gate_history.png", dpi=256)
        plt.close()

    def plot_epoch_layer_scores(self, epoch_layer_scores):
        """
        绘制每个 epoch 结束时各层重要性得分的变化趋势
        用于验证单任务内添加噪声的必要性（观察是否存在层偏置和先发优势）

        Args:
            epoch_layer_scores (List[ndarray]): 每个元素为 shape [num_layers]
        """
        import numpy as np
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'serif'
        scores_array = np.array(epoch_layer_scores)
        num_epochs, num_layers = scores_array.shape

        plt.figure(figsize=(14, 7))
        colors = plt.cm.get_cmap("tab20", num_layers)

        for layer_id in range(num_layers):
            layer_scores = scores_array[:, layer_id]
            plt.plot(range(num_epochs), layer_scores,
                    label=f'Layer {layer_id}',
                    color=colors(layer_id),
                    linewidth=2,
                    marker='o',
                    markersize=4)

            plt.text(num_epochs - 0.5, layer_scores[-1], f"L{layer_id}",
                    fontsize=9, color=colors(layer_id),
                    verticalalignment='center', fontweight='bold')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Layer Importance Score', fontsize=12)
        # plt.title(f'Layer Importance Evolution (Task {self._cur_task})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 均匀分布参考线
        uniform_score = 1.0 / num_layers
        plt.axhline(y=uniform_score, color='gray', linestyle='--',
                   linewidth=1.5, label=f'Uniform ({uniform_score:.3f})')

        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=8)
        plt.tight_layout()

        save_path = f"{self.dataset_name}_layer_scores_epoch_task{self._cur_task}_seed{self.seed}_{self.dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved: {save_path}")
        plt.close()

    def plot_task_layer_heatmap(self):
        """
        绘制所有任务的层重要性热力图
        横轴：Task 0-N
        纵轴：Layer 0-11
        颜色：层重要性得分
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置字体为 Times New Roman
        plt.rcParams['font.family'] = 'serif'

        # 转换为 numpy 数组：[num_tasks, num_layers]
        scores_array = np.array(self.task_layer_scores)
        num_tasks, num_layers = scores_array.shape

        # 转置使得层在纵轴，任务在横轴：[num_layers, num_tasks]
        scores_array = scores_array.T

        fig, ax = plt.subplots(figsize=(max(10, num_tasks * 0.8), 8))

        # 使用 seaborn 绘制热力图，去掉格子间的线条
        sns.heatmap(scores_array,
                    annot=True,  # 显示数值
                    fmt='.3f',   # 数值格式
                    cmap='YlOrRd',  # 颜色映射
                    cbar_kws={'label': 'Layer Importance Score'},
                    xticklabels=[f'{i}' for i in range(num_tasks)],
                    yticklabels=[f'L{i}' for i in range(num_layers)],
                    linewidths=0,  # 去掉格子间的线条
                    ax=ax)

        # 去掉外边框
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.xlabel('Task Sequence', fontsize=14)
        plt.ylabel('Layer Index', fontsize=14)
        # plt.title('Layer Importance Across Tasks', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = f"layer_importance_heatmap_seed{self.seed}_{self.dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Task-layer heatmap saved: {save_path}")
        plt.close()

    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            task_id = class_idx // self.inc
            self.cls2task[class_idx] = task_id

            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size*3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                output,_ = model(_inputs.to(self._device), train = False)
                _vectors = output["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            # 计算类原型（均值）
            class_mean = vectors.mean(dim=0).to(self._device)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = class_mean
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = class_mean
                self.cls_cov[class_idx] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device))
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"] # 10
                features_per_cls = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_lables == i]
                    cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)

                self.cls_mean[class_idx] = cluster_means
                self.cls_cov[class_idx] = cluster_vars

    def classifer_align(self, model):
        model.train()

        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'lapromptPool' not in n and 'task_identifiers' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)

                    # 检查是否包含 NaN 或 inf
                    if torch.isnan(mean).any() or torch.isinf(mean).any():
                        logging.warning(f"Class {class_idx} mean contains NaN or Inf, skipping")
                        continue
                    if torch.isnan(cov).any() or torch.isinf(cov).any():
                        logging.warning(f"Class {class_idx} cov contains NaN or Inf, skipping")
                        continue

                    # 添加小扰动以确保协方差矩阵正定
                    cov = cov + 1e-4 * torch.eye(cov.shape[0], device=cov.device)

                    try:
                        m = MultivariateNormal(mean.float(), cov.float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
                    except Exception as e:
                        logging.warning(f"Failed to sample for class {class_idx}: {str(e)}")
                        continue

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        try:
                            m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                            sampled_data.append(sampled_data_single)
                            sampled_label.extend([class_idx] * num_sampled_pcls)
                        except Exception as e:
                            logging.warning(f"Failed to sample for class {class_idx} cluster {cluster}: {str(e)}")
                            continue
            else:
                raise NotImplementedError

            if len(sampled_data) == 0:
                logging.error("No valid samples generated, skipping CA epoch")
                continue

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                # 为CA训练的合成样本确定任务ID
                class_ids = tgt[0].item()  # 该batch的类别
                task_id = class_ids // self.inc  # 根据类别计算任务ID

                outputs = model(inp, task_ids=task_id, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    # logging.warning(f"NaN or Inf detected in logits for class {class_ids}, skipping")
                    continue

                loss = F.cross_entropy(logits, tgt)

                if torch.isnan(loss) or torch.isinf(loss):
                    # logging.warning(f"NaN or Inf loss detected for class {class_ids}, skipping")
                    continue

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            avg_loss = losses / self._total_classes if self._total_classes > 0 else 0.0
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                avg_loss,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
