import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear
from einops.layers.torch import Reduce
from einops import rearrange

class perceiver_i_multi(nn.Module):
    def __init__(self,**kwargs):
        super(perceiver_i_multi, self).__init__()
        
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.cross_head = kwargs['cross_head']
        self.cross_dim = kwargs['cross_dim']
        self.self_head = kwargs['self_head']
        self.self_dim = kwargs['self_dim']
        self.n_layers = kwargs['n_layers']
        self.num_latents = kwargs['num_latents']
        self.dropout_ratio = kwargs['dropout_ratio']
        self.batch = kwargs['batch']
        self.device = kwargs['device']


        self.cross_attention_layer = torch.nn.MultiheadAttention(self.cross_dim, self.cross_head, dropout = self.dropout_ratio, batch_first=True)
        self.self_attention_layer = torch.nn.MultiheadAttention(self.self_dim, self.self_head, dropout = self.dropout_ratio, batch_first=True)

        self.mlp_cr = torch.nn.Sequential(
            nn.LayerNorm(self.cross_dim),
            nn.Linear(self.cross_dim, 2 * self.cross_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * self.cross_dim, self.cross_dim, bias=True),
        )


        self.mlp_sl = torch.nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, 2 * self.self_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * self.self_dim, self.self_dim, bias=True),
        )

        self.cr_sl = nn.Linear(self.cross_dim, self.self_dim)

        self.embedding = nn.Linear(self.fea, self.cross_dim)

        # self.to_logits = torch.nn.Sequential(
        #                                     Reduce('b n d -> b d', 'mean'),
        #                                     nn.LayerNorm(self.self_dim),
        #                                     nn.Linear(self.self_dim,self.cla))
        self.to_logits = torch.nn.Sequential(
                                    
                                    nn.LayerNorm(self.self_dim*self.num_latents),
                                    nn.Linear(self.self_dim*self.num_latents, 512),
                                    nn.GELU(),

                                    )
        
        self.to_logits_1 = torch.nn.Sequential(

                                    nn.Linear(512, 256),
                                    nn.GELU(),
                                    nn.Linear(256, 128),
                                    nn.GELU(),
                                    nn.Linear(128, 1)
                                    )
        self.to_logits_2 = torch.nn.Sequential(

                                    nn.Linear(512, 256),
                                    nn.GELU(),
                                    nn.Linear(256, 128),
                                    nn.GELU(),
                                    nn.Linear(128, 3)
                                    )
        self.to_logits_3 = torch.nn.Sequential(

                                    nn.Linear(512, 256),
                                    nn.GELU(),
                                    nn.Linear(256, 128),
                                    nn.GELU(),
                                    nn.Linear(128, 3)
                                    )
        self.to_logits_4 = torch.nn.Sequential(

                                    nn.Linear(512, 256),
                                    nn.GELU(),
                                    nn.Linear(256, 128),
                                    nn.GELU(),
                                    nn.Linear(128, 1)
                                    )

        self.layer_norm_cross = nn.LayerNorm(self.cross_dim)
        self.layer_norm_cross_latents = nn.LayerNorm(self.cross_dim)

        self.layer_norm_self = nn.LayerNorm(self.self_dim)

        self.dropout = nn.Dropout(self.dropout_ratio)   

        self.latents = nn.Parameter(torch.randn(self.num_latents, self.cross_dim)).to(self.device)

        self.log_sigma1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_sigma2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_sigma3 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_sigma4 = nn.Parameter(torch.zeros(1), requires_grad=True)

        



    def forward(self, data,pos,padding_index = None):
        if padding_index is not None:
            
            key_padding_mask = padding_index
        else:
            key_padding_mask = None
        x, pos = data, pos

        
        fea = torch.cat([x,pos],dim=2)

        batch_size = fea.shape[0]

        latents = self.latents.repeat(batch_size,1,1)



        fea = self.embedding(fea)



        latents = self.layer_norm_cross_latents(latents)
        out = self.layer_norm_cross(fea)
        
        out, _ = self.cross_attention_layer(latents, out, out, key_padding_mask=key_padding_mask)
        out = self.layer_norm_cross(self.dropout(out)  + latents)
        out  = self.dropout(self.mlp_cr(out)) + out
        out = self.layer_norm_cross(out)

        out = self.cr_sl(out)
        
        for i in range(self.n_layers):

            out = self.layer_norm_self(out)
            _out, _ = self.self_attention_layer(out, out, out)
            out = self.layer_norm_self(self.dropout(_out) + out)
            out = self.dropout(self.mlp_sl(out))+out
            out = self.layer_norm_self(out)

        out = rearrange(out, 'b n d -> b (n d)')
        out = self.to_logits(out)
        out1 = self.to_logits_1(out)
        out2 = self.to_logits_2(out)
        out3 = self.to_logits_3(out)
        out4 = self.to_logits_4(out)
            
        return out1, out2, out3, out4
