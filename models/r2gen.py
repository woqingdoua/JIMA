import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
import torch.nn.functional as F
#from modules.base_cmn import BaseCMN
from modules.wcl_projection import Projection
#from modules.encoder_decoder import EncoderDecoder
from modules.wcl_encoder_decoder import EncoderDecoder
from modules.rl_cmn import BaseCMN

class DistMemModel(nn.Module):
    def __init__(self,tokenizer):
        super(DistMemModel, self).__init__()
        self.rnn = nn.GRUCell(len(tokenizer.idx2token)+1, len(tokenizer.idx2token)+1)
        self.token_dis = nn.Linear(2048, len(tokenizer.idx2token)+1)
        self.ll = nn.Linear(len(tokenizer.idx2token)+1,len(tokenizer.idx2token)+1)

    def forward(self,prediction,fc_feats):

        #hx = torch.zeros_like(token_dist).cuda(0).unsqueeze(1)
        token_dis1 = self.token_dis(fc_feats)
        token_dis =  token_dis1
        output = []
        for i in range(prediction.size()[1]):
            token_dist = self.rnn(prediction[:,i], token_dis)
            token_dist_ = token_dist.unsqueeze(1)
            output.append(token_dist_)
        output1 = F.sigmoid(torch.stack(output, dim=1).squeeze(2)) * prediction
        #output = F.log_softmax(output,dim=-1)

        return output1,token_dis1

    def next(self,prediction,token_dist=None):

        token_dist = self.rnn(prediction, token_dist)
        output = F.sigmoid(token_dist) * prediction
        #output = F.log_softmax(output,dim=-1)
        return output, token_dist




class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        #self.encoder_decoder = BaseCMN(args, tokenizer)
        #self.projection = Projection(args.contra_embed_size)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

        #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        #self.cls = nn.Sequential(nn.Linear(512,100),nn.ReLU(),nn.Linear(100,1))

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def normalization(self,x):
        x = F.softmax(x,dim=0)
        x = x*len(x)
        x = x + (1. - torch.mean(x))
        return x

    def forward_iu_xray(self, images, targets=None, mode='train'):


        if mode == 'pretrain':
            word_embed = self.encoder_decoder.model.tgt_embed(targets)
            word_represent = self.transformer_encoder(word_embed)
            pre_label = F.sigmoid(self.cls(torch.mean(word_represent,dim=1)))
            return pre_label

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            #output,token_dis = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output,fc_feats

        elif mode == 'sample':
            output, prob = self.encoder_decoder(fc_feats, att_feats,distmem=None, mode='sample')
            return output,prob,fc_feats
        else:
            raise ValueError


    def forward_mimic_cxr(self, images, targets=None,distmem=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output,fc_feats
        elif mode == 'sample':
            token_dist = distmem.token_dis(fc_feats)
            output, prob = self.encoder_decoder(fc_feats, att_feats, mode='sample',distmem=distmem,token_dist=token_dist)
            return output,prob,fc_feats
        else:
            raise ValueError


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            try:
                #(4,4096), (4,98.2048)
                output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            except:
                pass
            return output, output_probs
        else:
            raise ValueError
        # return output

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
        # return output

