import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, EaseCosineLinear, SimpleContinualLinear
import timm
from torch.nn import functional as F

checkpoint_path_in1k = "/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
checkpoint_path_in21k = "/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
# checkpoint_path_in21k_orig = "/pretrains/ViT-B_16.npz"

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if args["model_name"] == "liprompt":
        from backbone import vit_lieprompt
        if "in21k" in name:
            checkpoint_path = checkpoint_path_in21k
        else:
            checkpoint_path = checkpoint_path_in1k

        model = timm.create_model(
            args["backbone_type"],
            pretrained=args["pretrained"],
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
            head_type = args["head_type"],
            prompt_length=args["length"],
            top_k=args["top_k"],
            use_prefix_tune_for_prompt=args["use_prefix_tune_for_prompt"],
            pool_size=args["pool_size"],
            prompt_pool=args["prompt_pool"],
            use_prompt_key=args["use_prompt_key"],
            prompt_key_init=args["prompt_key_init"],
            checkpoint_path = checkpoint_path
        )
        return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class LiEpromptNet(BaseNet):
    def __init__(self, args, pretrained):
        super(LiEpromptNet, self).__init__(args, pretrained)
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.task_indicator = None
        self.all_layer_scores = []
        self._cur_task = -1
        self.fc = None
        self.out_dim =  768

    @property
    def feature_dim(self):
        return 768
    
    def generate_indicator(self, in_dim, out_dim):
        task_indicator = CosineLinear(in_dim, out_dim)
        return task_indicator
    
    def update_task_indicator(self, nb_classes, nextperiod_initialization=None):
        feature_dim = 768
        task_indicator = self.generate_indicator(feature_dim, nb_classes)
        if self.task_indicator is not None:
            old_nb_classes = self.task_indicator.out_features
            weight = copy.deepcopy(self.task_indicator.weight.data)
            task_indicator.sigma.data = self.task_indicator.sigma.data
            task_indicator.weight.data[ : old_nb_classes, :] = nn.Parameter(weight)
        del self.task_indicator
        self.task_indicator = task_indicator
    
    def compute_layer_importance(self):
        avg_layer_score = self.backbone.RouterLinear.get_running_score()
        self.all_layer_scores.append(avg_layer_score)
        return avg_layer_score
        
    def forward(self, x, train=False, fc_only=False):
        original_feature = self.backbone.forward_original(x)  # [B, ..., D]
        if train:
            output = self.backbone(x, original_feature, train=train, fc_only=fc_only)
            return output
        else:
            proto_logits = self.task_indicator(original_feature)["logits"]
            predicts = torch.topk(
                    proto_logits, k=1, dim=1, largest=True, sorted=True
                )[
                    1
                ]  # [bs, topk]
            task_ids = (predicts - self.init_cls) // self.inc + 1
            output = self.backbone(x, original_feature, task_ids=task_ids, train=train, fc_only=fc_only)
            return output, proto_logits