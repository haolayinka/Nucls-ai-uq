#!/usr/bin/env python3
"""CPU-compatible architecture smoke test, reconstructed from official HoVer-Net
MIT-licensed source, vqdang/hover_net, models/hovernet/net_desc.py,
models/hovernet/net_utils.py and models/hovernet/utils.py (accessed 2026-09-24).

NOT the unmodified official inference runner. Does not implement official tiling,
instance postprocessing, detection confidence, validation, or AI performance.
"""
import argparse, hashlib, json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

CHECKPOINT_SHA256='5d1191d6bf72a077911aef12a75484ee3cab91ed81e7880c0361157094ff451d'

def crop_op(x, crop):
    t,l=crop[0]//2,crop[1]//2
    b,r=crop[0]-t,crop[1]-l
    return x[:, :,t:x.shape[2]-b,l:x.shape[3]-r]

def crop_to_shape(x,y):
    return crop_op(x,(x.shape[2]-y.shape[2],x.shape[3]-y.shape[3]))

class SamePad(nn.Module):
    def __init__(self,ksize,stride):
        super().__init__();self.ksize=ksize;self.stride=stride
    def forward(self,x):
        v=x.shape[2]%self.stride
        pad=max(self.ksize-(self.stride if v==0 else v),0)
        start=pad//2
        return F.pad(x,(start,pad-start,start,pad-start))

class DenseBlock(nn.Module):
    def __init__(self,in_ch,ksizes,channels,count,split=1):
        super().__init__();self.units=nn.ModuleList()
        for i in range(count):
            self.units.append(nn.Sequential(OrderedDict([
                ('preact_bna/bn',nn.BatchNorm2d(in_ch+i*channels[1],eps=1e-5)),
                ('preact_bna/relu',nn.ReLU(inplace=True)),
                ('conv1',nn.Conv2d(in_ch+i*channels[1],channels[0],ksizes[0],bias=False)),
                ('conv1/bn',nn.BatchNorm2d(channels[0],eps=1e-5)),
                ('conv1/relu',nn.ReLU(inplace=True)),
                ('conv2',nn.Conv2d(channels[0],channels[1],ksizes[1],groups=split,bias=False)),
            ])))
        self.blk_bna=nn.Sequential(OrderedDict([
            ('bn',nn.BatchNorm2d(in_ch+count*channels[1],eps=1e-5)),
            ('relu',nn.ReLU(inplace=True))]))
    def forward(self,x):
        for unit in self.units:
            new=unit(x);x=torch.cat([crop_to_shape(x,new),new],dim=1)
        return self.blk_bna(x)

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,ksizes,channels,count,stride=1):
        super().__init__();self.units=nn.ModuleList()
        for i in range(count):
            ch0=in_ch if i==0 else channels[-1]
            step=stride if i==0 else 1
            layers=[
                ('preact/bn',nn.BatchNorm2d(ch0,eps=1e-5)),
                ('preact/relu',nn.ReLU(inplace=True)),
                ('conv1',nn.Conv2d(ch0,channels[0],ksizes[0],bias=False)),
                ('conv1/bn',nn.BatchNorm2d(channels[0],eps=1e-5)),
                ('conv1/relu',nn.ReLU(inplace=True)),
                ('conv2/pad',SamePad(ksizes[1],step)),
                ('conv2',nn.Conv2d(channels[0],channels[1],ksizes[1],stride=step,bias=False)),
                ('conv2/bn',nn.BatchNorm2d(channels[1],eps=1e-5)),
                ('conv2/relu',nn.ReLU(inplace=True)),
                ('conv3',nn.Conv2d(channels[1],channels[2],ksizes[2],bias=False)),
            ]
            self.units.append(nn.Sequential(OrderedDict(layers if i else layers[2:])))
        self.shortcut=nn.Conv2d(in_ch,channels[-1],1,stride=stride,bias=False) if in_ch!=channels[-1] or stride!=1 else None
        self.blk_bna=nn.Sequential(OrderedDict([
            ('bn',nn.BatchNorm2d(channels[-1],eps=1e-5)),
            ('relu',nn.ReLU(inplace=True))]))
    def forward(self,x):
        shortcut=x if self.shortcut is None else self.shortcut(x)
        for unit in self.units:
            x=unit(x)+shortcut
            shortcut=x
        return self.blk_bna(x)

class UpSample2x(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('unpool_mat',torch.ones((2,2),dtype=torch.float32))
    def forward(self,x):
        return x.repeat_interleave(2,dim=2).repeat_interleave(2,dim=3)

class HoVerNetOriginal(nn.Module):
    def __init__(self,nr_types=5):
        super().__init__();self.conv0=nn.Sequential(OrderedDict([
            ('/',nn.Conv2d(3,64,7,bias=False)),
            ('bn',nn.BatchNorm2d(64,eps=1e-5)),('relu',nn.ReLU(inplace=True))]))
        self.d0=ResidualBlock(64,[1,3,1],[64,64,256],3,1)
        self.d1=ResidualBlock(256,[1,3,1],[128,128,512],4,2)
        self.d2=ResidualBlock(512,[1,3,1],[256,256,1024],6,2)
        self.d3=ResidualBlock(1024,[1,3,1],[512,512,2048],3,2)
        self.conv_bot=nn.Conv2d(2048,1024,1,bias=False)
        def branch(out_ch):
            def dec(in_ch,conv_ch,count,out_ch2):
                return nn.Sequential(OrderedDict([
                    ('conva',nn.Conv2d(in_ch,conv_ch,5,bias=False)),
                    ('dense',DenseBlock(conv_ch,[1,5],[128,32],count,split=4)),
                    ('convf',nn.Conv2d(conv_ch+32*count,out_ch2,1,bias=False)),
                ]))
            return nn.Sequential(OrderedDict([
                ('u3',dec(1024,256,8,512)),('u2',dec(512,128,4,256)),
                ('u1',nn.Sequential(OrderedDict([
                    ('conva/pad',SamePad(5,1)),('conva',nn.Conv2d(256,64,5,bias=False))]))),
                ('u0',nn.Sequential(OrderedDict([
                    ('bn',nn.BatchNorm2d(64,eps=1e-5)),('relu',nn.ReLU(inplace=True)),
                    ('conv',nn.Conv2d(64,out_ch,1,bias=True))]))),
            ]))
        self.decoder=nn.ModuleDict(OrderedDict([('tp',branch(nr_types)),('np',branch(2)),('hv',branch(2))]))
        self.upsample2x=UpSample2x()
    def forward(self,imgs):
        d0=self.d0(self.conv0(imgs/255.0))
        d1=self.d1(d0)
        d2=self.d2(d1)
        d3=self.conv_bot(self.d3(d2))
        d0=crop_op(d0,(184,184));d1=crop_op(d1,(72,72))
        out=OrderedDict()
        for name,desc in self.decoder.items():
            u3=desc[0](self.upsample2x(d3)+d2)
            u2=desc[1](self.upsample2x(u3)+d1)
            u1=desc[2](self.upsample2x(u2)+d0)
            out[name]=desc[3](u1)
        return out

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='/mnt/data/hovernet_original_consep_type_tf2pytorch.tar')
    ap.add_argument('--image',default=None);ap.add_argument('--output',default='/mnt/data/nucls_hovernet_cpu_smoke_v10.json')
    args=ap.parse_args()
    torch.set_num_threads(4)
    assert sha256(args.checkpoint)==CHECKPOINT_SHA256,'checkpoint bytes differ'
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['desc']
    model=HoVerNetOriginal(nr_types=5)
    keys=model.load_state_dict(weights,strict=True)
    print('STRICT_LOAD_PASS',len(weights),'tensors; result',keys)
    result={'checkpoint_sha256':CHECKPOINT_SHA256,'strict_load':True,'tensor_count':len(weights),
            'architecture':'CPU reconstructed from official source; NOT unchanged official runner',
            'status':'ARCHITECTURE_ONLY_UNTIL_IMAGE_FORWARD_COMPLETES'}
    if args.image:
        im=np.array(Image.open(args.image).convert('RGB'))
        # Source image-only smoke patch; NO scoring key read, NO labels loaded.
        # Fixed central 270x270 patch in original source pixel space, not proposal-resampled.
        h,w=im.shape[:2]
        if min(h,w)<270:
            raise ValueError('selected image too small for a native 270x270 patch')
        left=(w-270)//2;top=(h-270)//2
        patch=im[top:top+270,left:left+270]
        x=torch.from_numpy(patch.copy()).permute(2,0,1).unsqueeze(0).float()
        model.eval()
        with torch.inference_mode():out=model(x)
        for name,t in out.items():
            assert tuple(t.shape)==(1,{'tp':5,'np':2,'hv':2}[name],80,80),(name,t.shape)
            assert torch.isfinite(t).all(),name
        result.update({'status':'RAW_NETWORK_PATCH_FORWARD_PASS_NOT_INSTANCE_INFERENCE',
              'image_basename':Path(args.image).name,'image_sha256':sha256(args.image),
              'input_shape':[270,270,3],'source_crop_left_top':[left,top],
              'raw_logit_shapes':{k:list(v.shape) for k,v in out.items()},
              'nuclear_probability_min_max':[float(v) for v in (torch.softmax(out['np'],dim=1)[:,1].min(),torch.softmax(out['np'],dim=1)[:,1].max())],
              'no_instance_postprocessing':True,'no_scoring':True})
        print('IMAGE_FORWARD_PASS',result['raw_logit_shapes'])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('OUTPUT',args.output)
if __name__=='__main__':main()
