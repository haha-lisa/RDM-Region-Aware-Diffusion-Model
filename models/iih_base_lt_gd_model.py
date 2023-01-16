import torch
import os
import itertools
import torch.nn.functional as F
from util import distributed as du
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
import util.ssim as ssim

class IIHBaseLTGDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='base_lt_gd', dataset_mode='ihd')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_R_gradient', type=float, default=20., help='weight for reflectance gradient loss')
            parser.add_argument('--lambda_I_L2', type=float, default=10, help='weight for illumination L2 loss')
            parser.add_argument('--lambda_I_smooth', type=float, default=1, help='weight for Illumination smooth loss')
            parser.add_argument('--lambda_ifm', type=float, default=100, help='weight for pm loss')
            
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['G','G_L1','G_R_grident','G_I_L2','G_I_smooth',"IF"]
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['mask', 'harmonized','comp','real','reflectance','illumination','ifm_mean']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        self.cur_device = torch.cuda.current_device()
        self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        if self.ismaster:
            print(self.netG)  
        
        if self.isTrain:
            if self.ismaster == 0:
                util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss().cuda(self.cur_device)
            self.criterionL2 = torch.nn.MSELoss().cuda(self.cur_device)
            self.criterionDSSIM_CS = ssim.DSSIM(mode='c_s').to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']

        self.mask_r = F.interpolate(self.mask, size=[64,64])
        self.mask_r_32 = F.interpolate(self.mask, size=[32,32])
        self.real_r = F.interpolate(self.real, size=[32,32])
        self.real_gray = util.rgbtogray(self.real_r)
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.harmonized, self.reflectance, self.illumination, self.ifm_mean = self.netG(self.inputs, self.mask_r, self.mask_r_32)
        if not self.isTrain:
            self.harmonized = self.comp*(1-self.mask) + self.harmonized*self.mask
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_IF = self.criterionDSSIM_CS(self.ifm_mean, self.real_gray)*self.opt.lambda_ifm

        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G_R_grident = self.gradient_loss(self.reflectance, self.real)*self.opt.lambda_R_gradient
        self.loss_G_I_L2 = self.criterionL2(self.illumination, self.real)*self.opt.lambda_I_L2
        self.loss_G_I_smooth = util.compute_smooth_loss(self.illumination)*self.opt.lambda_I_smooth
        # assert 0
        self.loss_G = self.loss_G_L1 + self.loss_G_R_grident + self.loss_G_I_L2 + self.loss_G_I_smooth + self.loss_IF
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y
   
