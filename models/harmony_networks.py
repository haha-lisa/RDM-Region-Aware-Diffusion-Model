import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from util import util
from . import networks as networks_init


###############################################################################
# Helper Functions
###############################################################################

def define_G(netG='base',init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    if netG == 'base':
        net = BaseGenerator(opt)
    elif netG == 'base_lt':
        net = BaseLTGenerator(opt)
    elif netG == 'base_gd':
        net = BaseGDGenerator(opt)
    elif netG == 'base_lt_gd':
        net = BaseLTGDGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)
    return net

##############################################################################
# Classes
##############################################################################
class BaseGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 4
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

    def forward(self, inputs, mask=None):
        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)
        reflectance = self.reflectance_dec(r_content)
        
        illumination = self.illumination_dec(i_content)
        
        reflectance = reflectance / 2 +0.5
        illumination = illumination / 2 + 0.5
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination

class BaseLTGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseLTGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 0
        i_dec_n_res = 0

        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)
       
        self.lighting = GlobalLighting(opt.n_downsample, opt.input_nc+1, opt.ngf, opt.light_mlp_dim, 'none', opt.activ, pad_type=opt.pad_type)
        self.lightingRes = LightingResBlocks(opt.illumination_n_res, self.illumination_enc.output_dim, opt.light_mlp_dim, norm='ln', activation=opt.activ, pad_type=opt.pad_type)
        

    def forward(self, inputs, mask=None, mask_r=None):
        
        l_fg, l_bg = self.lighting(inputs, mask_r)

        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)

        reflectance = self.reflectance_dec(r_content)
        reflectance = reflectance / 2 +0.5

        i_content = self.lightingRes(i_content, l_fg, l_bg, mask_r)
        illumination = self.illumination_dec(i_content)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination

class BaseGDGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseGDGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 4
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.ifm = InharmonyfreeCAModule(opt.n_downsample+1, opt.ifm_n_res, opt.input_nc+1, self.reflectance_enc.output_dim, opt.ngf//2, opt.inharmonyfree_norm, opt.activ, pad_type=opt.pad_type)

        self.reflectanceRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.reflectance_enc.output_dim)
        self.illuminationRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.illumination_enc.output_dim)


    def forward(self, inputs, mask_r, mask_r_32=None):

        match_score, ifm_mean = self.ifm(inputs, mask_r_32)

        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)

        r_content = self.reflectanceRec(r_content, fg_mask=mask_r, attScore=match_score)
        i_content = self.illuminationRec(i_content, fg_mask=mask_r, attScore=match_score.detach())

        reflectance = self.reflectance_dec(r_content)
        reflectance = reflectance / 2 +0.5

        illumination = self.illumination_dec(i_content)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination, ifm_mean

class BaseLTGDGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseLTGDGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 0
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc+1, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)
       
        self.lighting = GlobalLighting(opt.n_downsample, opt.input_nc+1, opt.ngf, opt.light_mlp_dim, 'none', opt.activ, pad_type=opt.pad_type)
        self.lightingRes = LightingResBlocks(opt.illumination_n_res, self.illumination_enc.output_dim, opt.light_mlp_dim, norm='ln', activation=opt.activ, pad_type=opt.pad_type)

        self.ifm = InharmonyfreeCAModule(opt.n_downsample+1, opt.ifm_n_res, opt.input_nc+1, self.reflectance_enc.output_dim, opt.ngf//2, opt.inharmonyfree_norm, opt.activ, pad_type=opt.pad_type)

        self.reflectanceRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.reflectance_enc.output_dim)
        self.illuminationRec = HarmonyRecBlocks(opt.inharmonyfree_embed_layers, dim=self.illumination_enc.output_dim)
    def forward(self, inputs, mask_r=None, mask_r_32=None):
        fg_pooling, bg_pooling= self.lighting(inputs, mask_r)
        # match_score, ifm_mean = self.ifm(inputs, mask_r_32, self.lamda)
        match_score, ifm_mean = self.ifm(inputs, mask_r_32)

        r_content = self.reflectance_enc(inputs)
        i_content = self.illumination_enc(inputs)
        
        r_content = self.reflectanceRec(r_content, fg_mask=mask_r, attScore=match_score)
        
        reflectance = self.reflectance_dec(r_content)
        reflectance = reflectance / 2 +0.5

        i_content = self.lightingRes(i_content, fg_pooling, bg_pooling, mask_r)
        i_content = self.illuminationRec(i_content, fg_mask=mask_r, attScore=match_score.detach())
        
        illumination = self.illumination_dec(i_content)
        illumination = illumination / 2 + 0.5
        
        harmonized = reflectance*illumination

        return harmonized, reflectance, illumination, ifm_mean

      
##################################################################################
# Encoder and Decoders
##################################################################################

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentDecoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentDecoder, self).__init__()
        self.model = []
        dim = input_dim
        # upsampling blocks
        for i in range(n_downsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
            ]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class GlobalLighting(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, light_mlp_dim=8, norm=None, activ=None, pad_type='zero'):
    
        super(GlobalLighting, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        # self.model += [ResBlocks(4, dim, norm=norm, activation=activ, pad_type=pad_type)]
        # if not dim == output_dim:
        #     self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.light_mlp = LinearBlock(dim, light_mlp_dim, norm='none', activation='none')

    def forward(self, x, fg_mask):
        x = self.model(x)
        b,c,h,w = x.size()

        fg_mask_sum = torch.sum(fg_mask.view(b, 1, -1), dim=2)
        bg_mask_sum = h*w - fg_mask_sum+1e-8
        fg_mask_sum = fg_mask_sum +1e-8
        x_bg = x*(1-fg_mask)

        # avg pooling to 1*1
        x_bg_pooling = torch.sum(x_bg.view(b,c,-1), dim=2).div(bg_mask_sum)
        l_bg = self.light_mlp(x_bg_pooling)

        x_fg = x*fg_mask
        x_fg_pooling = torch.sum(x_fg.view(b,c,-1), dim=2).div(fg_mask_sum)
        l_fg = self.light_mlp(x_fg_pooling)

        return l_fg, l_bg
    
class InharmonyfreeCAModule(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type, lamda=10):
        super(InharmonyfreeCAModule, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 3,1,1, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.in_normal = nn.InstanceNorm1d(1024)

    def forward(self, x, fg_mask=None,lamda=10):
        content = self.model(x)
        b,c,h,w = content.size()

        fg = content*fg_mask
        bg = content*(1-fg_mask)
        fg_patch = fg.view(b,c,-1).permute(0,2,1)
        bg_patch = bg.view(b,c,-1)

        fg_patch_mu = torch.mean(fg_patch, dim=2, keepdim=True)
        bg_patch_mu = torch.mean(bg_patch, dim=1, keepdim=True)
        fg_bg_conv = torch.matmul((fg_patch-fg_patch_mu), (bg_patch-bg_patch_mu))/(c-1)
        
        match_score_soft = F.softmax(lamda * fg_bg_conv.permute(0,2,1), dim=1)
        match_score_soft = match_score_soft.view(b, -1, h, w)
        content_mean = torch.mean(content, dim=1, keepdim=True)
        return match_score_soft, content_mean

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class LightingResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, light_mlp_dim=8, norm='in', activation='relu', pad_type='zero'):
        super(LightingResBlocks, self).__init__()
        self.resblocks = nn.ModuleList([LightingResBlock(dim, light_mlp_dim, norm=norm, activation=activation, pad_type=pad_type) for i in range(num_blocks)])

    def forward(self, x, fg, bg, fg_mask):
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, fg, bg, fg_mask)

        return x

class HarmonyRecBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(HarmonyRecBlocks, self).__init__()
        self.resblocks = nn.ModuleList([HarmonyRecBlock(stride=1, rate=2, channels=dim) for i in range(num_blocks)])

    def forward(self, x, fg_mask=None, attScore=None):
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, fg_mask, attScore)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, input_dim//2, norm=norm, activation=activ)]
        dim = input_dim//2
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim//2, norm=norm, activation=activ)]
            dim = dim//2
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class LightingResBlock(nn.Module):
    def __init__(self, dim, light_mlp_dim, norm='in', activation='relu', pad_type='zero'):
        super(LightingResBlock, self).__init__()
        
        self.lt_1 = Lighting(dim, light_mlp_dim, norm=norm, activation=activation, pad_type=pad_type)
        self.conv_1 = nn.Conv2d(dim, dim, 3, 1)
        self.lt_2 = Lighting(dim, light_mlp_dim, norm=norm, activation=activation, pad_type=pad_type)
        self.conv_2 = nn.Conv2d(dim, dim, 3, 1)
        self.norm = LayerNorm(dim)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x, fg, bg, fg_mask):
        residual = x
        out_1 = self.actvn(self.norm(self.lt_1(self.conv_1(self.pad(x)), fg, bg, fg_mask)))
        out = self.norm(self.lt_2(self.conv_2(self.pad(out_1)), fg, bg, fg_mask))
        out += residual
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class Lighting(nn.Module):
    def __init__(self, dim, light_mlp_dim, norm='ln', activation='relu', pad_type='zero'):
        super(Lighting, self).__init__()

        self.light_mlp = LinearBlock(light_mlp_dim, 4*dim, norm='none', activation=activation)
        self.rgb_model = Conv2dBlock(dim ,dim*3, 3, 1, 1, norm='grp', activation=activation, pad_type=pad_type, groupcount=3)

        self.dim = dim

    def forward(self, x, fg, bg, fg_mask):
        residual = x
        b,c,h,w = x.size()
        illu_fg_color_ratio, illu_fg_intensity = self.illumination_extract(fg)
        illu_bg_color_ratio, illu_bg_intensity = self.illumination_extract(bg)
        
        illu_color_ratio = illu_bg_color_ratio.div(illu_fg_color_ratio+1e-8)
        illu_intensity = illu_bg_intensity - illu_fg_intensity
        b,c,h,w = x.size()

        x_rgb = self.rgb_model(x)
        x_rgb = x_rgb.view(b,3,c,h, w)
        illu_color_ratio = illu_color_ratio.view(b, 3, c, 1, 1).expand_as(x_rgb)
        x_t_c = torch.sum(x_rgb*illu_color_ratio, dim=1)

        illu_intensity = illu_intensity.view(b,c,1,1).expand_as(x)

        x_t = x_t_c + illu_intensity
        
        output = residual*(1-fg_mask)+x_t*fg_mask
        return output

    
    def illumination_extract(self, x):
        b = x.size(0)
        illumination = self.light_mlp(x)
        illu_intensity = illumination[:, :self.dim]
        illu_color = illumination[:, self.dim:].view(b, 3, self.dim)
        illu_color_ratio = torch.softmax(illu_color,dim=1)
        return illu_color_ratio, illu_intensity

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class HarmonyRecBlock(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, channels=128):
        super(HarmonyRecBlock, self).__init__()
        self.ksize = ksize
        self.kernel = 2 * rate
        self.stride = stride
        self.rate = rate
        self.harmonyRecConv = Conv2dBlock(channels*2, channels, 3, 1, 1, norm='none', activation='relu', pad_type='reflect')

    def forward(self, bg_in, fg_mask=None, attScore=None):
        b, dims, h, w = bg_in.size()
        
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(bg_in, ksizes=[self.kernel, self.kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(b, dims, self.kernel, self.kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        ACL = []

        for ib in range(b):
            CA = attScore[ib:ib+1, :, :, :]
            k2 = raw_w[ib, :, :, :, :]
            ACLt = F.conv_transpose2d(CA, k2, stride=self.rate, padding=1)
            ACLt = ACLt / 4
            if ib == 0:
                ACL = ACLt
            else:
                ACL = torch.cat([ACL, ACLt], dim=0)
        con1 = torch.cat([bg_in, ACL], dim=1)
        ACL2 = self.harmonyRecConv(con1)
        return ACL2+bg_in

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

