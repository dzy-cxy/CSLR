import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
<<<<<<< Updated upstream
=======
from modules import Attention_Net, Spatial_Attention_Net, Temporal_Attention_Net

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
<<<<<<< Updated upstream
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
=======
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        
        self.resnet18 = models.resnet18(pretrained=True)#加载model
        #self.Attention_Net = Attention_Net.ResidualNet( 'ImageNet', 18, 1000, 'CBAM')#自定义网络
        self.Spatial_Attention_Net = Spatial_Attention_Net.ResidualNet( 'ImageNet', 18, 1000, 'CBAM')
        self.Temporal_Attention_Net = Temporal_Attention_Net.ResidualNet( 'ImageNet', 18, 1000, 'CBAM')

        #读取参数
        pretrained_dict = self.resnet18.state_dict()
        Spatial_model_dict = self.Spatial_Attention_Net.state_dict()
        Temporal_model_dict = self.Temporal_Attention_Net.state_dict()

        # 将pretrained_dict里不属于model_dict的键剔除掉
        Spatial_pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in Spatial_model_dict}
        Temporal_pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in Temporal_model_dict}

        # 更新现有的model_dict
        Spatial_model_dict.update(Spatial_pretrained_dict)
        Temporal_model_dict.update(Temporal_pretrained_dict)

        # 加载真正需要的state_dict
        self.Spatial_Attention_Net.load_state_dict(Spatial_model_dict)
        self.Temporal_Attention_Net.load_state_dict(Temporal_model_dict)
        self.Spatial_conv2d = self.Spatial_Attention_Net
        self.Temporal_conv2d = self.Temporal_Attention_Net
        
        self.Spatial_conv2d.fc = Identity()
        self.Temporal_conv2d.fc = Identity()
        
>>>>>>> Stashed changes
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def Spatial_masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.Spatial_conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x
    
    def Temporal_masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.Temporal_conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #print(batch, temp, channel, height, width)
            inputs = x.reshape(batch * temp, channel, height, width)
            Spatial_framewise = self.Spatial_masked_bn(inputs, len_x).reshape(batch, temp, -1).transpose(1, 2)
            Temporal_framewise = self.Temporal_masked_bn(inputs, len_x).reshape(batch, temp, -1).transpose(1, 2)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            Spatial_framewise = x
            Temporal_framewise = x

        Spatial_conv1d_outputs = self.conv1d(Spatial_framewise, len_x)
        Temporal_conv1d_outputs = self.conv1d(Temporal_framewise, len_x)
        
        # x: T, B, C
        Temporal_x = Temporal_conv1d_outputs['visual_feat']
        Temporal_lgt = Temporal_conv1d_outputs['feat_len']
        
        x = Spatial_conv1d_outputs['visual_feat']
        lgt = Spatial_conv1d_outputs['feat_len']
        
        tm_outputs = self.temporal_model(Temporal_x, Temporal_lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, Temporal_lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(Temporal_conv1d_outputs['conv_logits'], Temporal_lgt, batch_first=False, probs=False)

        return {
            "framewise_features": Temporal_framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": Temporal_conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
