
import torch
import logging
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

# Defining the BoQBlock, used to learn a set of global learned queries that probe local features
# This will enhance the cross-image encoder done by CricaVPRNet
class BoQBlock(nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()
        

class CricaVPRNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())

        # BoQ encoding
        in_dim = 768
        num_queries = 32
        num_layers = 2
        self.boqs = nn.ModuleList([BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])

        # In TransformerEncoderLayer, "batch_first=False" means the input tensors should be provided as (seq, batch, feature) to encode on the "seq" dimension.
        # Our input tensor is provided as (batch, seq, feature), which performs encoding on the "batch" dimension.
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # Cross-image encoder

    def forward(self, x):
        x = self.backbone(x)       

        B,P,D = x["x_prenorm"].shape
        print(f'B, P, D: {B}, {P}, {D}')
        #32, 257, 768
        W = H = int(math.sqrt(P-1))
        x0 = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 

        x10,x11,x12,x13 = self.aggregation(x_p[:,:,0:8,0:8]),self.aggregation(x_p[:,:,0:8,8:]),self.aggregation(x_p[:,:,8:,0:8]),self.aggregation(x_p[:,:,8:,8:])
        x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.aggregation(x_p[:,:,0:5,0:5]),self.aggregation(x_p[:,:,0:5,5:11]),self.aggregation(x_p[:,:,0:5,11:]),\
                                        self.aggregation(x_p[:,:,5:11,0:5]),self.aggregation(x_p[:,:,5:11,5:11]),self.aggregation(x_p[:,:,5:11,11:]),\
                                        self.aggregation(x_p[:,:,11:,0:5]),self.aggregation(x_p[:,:,11:,5:11]),self.aggregation(x_p[:,:,11:,11:])
        x = [i.unsqueeze(1) for i in [x0,x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]

        x = torch.cat(x,dim=1)

        #here the shape of x should be B x 14 x D
        logging.info(f'x Shape: {x.shape}')
        # 32 x 14 x 768
        print(x.shape)

        #assuming that it is...

        #BoQ encoding on the sequence
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
        out = torch.cat(outs, dim=1)

        logging.info(f'Out Shape: {out.shape}')
        # 32 x 64 x 768
        print(out.shape)

        #project back down to 32 x 14 x 768

        seq_projection = nn.Linear(64, 14).to('cuda')

        output = seq_projection(out.transpose(1, 2)).transpose(1, 2)

        logging.info(f'Output Shape: {output.shape}')
        print(output.shape)

        #Feed BoQ encoded outputs to cross-image encoder
        x = output

        #Cross-Image encoding on the batch
        x = self.encoder(x).view(B,14*D)

        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        logging.info(f'Final Shape: {x.shape}')
        print(x.shape)
        return x

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone

