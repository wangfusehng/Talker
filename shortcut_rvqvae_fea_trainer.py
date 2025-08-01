import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
from typing import Dict
from utils import config, logger_tools, other_tools, metric, data_transfer, other_tools_hf
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask,face_body_mask
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa
from models.vq.model import RVQVAE
import wandb


class CustomTrainer(train.BaseTrainer):
    '''
    Multi-Modal AutoEncoder
    '''
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        self.args = args
        self.cfg = cfg
        self.joints = self.train_data.joints

        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]#jaw，leye,reye
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
        
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self","predict_x0_loss"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False,False,False,False])
        
        ##### Model #####

        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        
        if args.ddp:
            self.model = getattr(model_module, cfg.model.g_name)(cfg).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)
        
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
        
        
        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        # Body part VQ models
        self.vq_models = self._create_body_vq_models()
        # print(self.vq_models)
        # Set all VQ models to eval mode
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        # if self.cfg.model.use_face_pose:
        #     self.vq_model_upper, self.vq_model_hands, self.vq_model_face , self.vq_model_lower= self.vq_models.values()
        # else:
        #     self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        self.vq_model_upper = self.vq_models['upper']
        self.vq_model_hands = self.vq_models['hands']
        if self.args.use_trans:
            self.vq_model_lower = self.vq_models['lower_trans']
        else:
            self.vq_model_lower = self.vq_models['lower']  # 或 'lower_trans'
        if self.cfg.model.use_face_pose:
            self.vq_model_face = self.vq_models['face'] 
        if self.cfg.model.use_exp:
            self.vq_model_exp = self.vq_models['exp']   
        self.vqvae_latent_scale = self.args.vqvae_latent_scale 

        self.args.vae_length = 240
        
        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        
        
        ##### Normalization #####
        self.use_trans = self.args.use_trans
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        
        # Extract body part specific normalizations
        for part in ['upper', 'hands', 'lower','face']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(self.mean[mask]).cuda())
            setattr(self, f'std_{part}', torch.from_numpy(self.std[mask]).cuda())
        
        # Translation normalization if needed
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).cuda()
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).cuda()
            
    def _create_body_vq_models(self) -> Dict[str, RVQVAE]:
        """Create VQ-VAE models for body parts."""
        vq_configs = {
            'upper': {'dim_pose': 78},
            'hands': {'dim_pose': 180},
            'lower': {'dim_pose': 54 if not self.args.use_trans else 57}
        }
        if  self.cfg.model.use_face_pose:
            vq_configs['face'] = {'dim_pose': 18}
        if self.args.use_trans:
            vq_configs.pop('lower')  # remove lower
            vq_configs['lower_trans'] = {'dim_pose': 57}  # lower pose + translation
        else:
            vq_configs['lower'] = {'dim_pose': 54}  # lower pose only
        if self.cfg.model.use_exp:
            vq_configs['exp'] = {'dim_pose': 100}
        vq_models = {}
        for part, config in vq_configs.items():
            print(f"Creating VQ-VAE model for {part} with config: {config}")
            model = self._create_rvqvae_model(config['dim_pose'], part)
            vq_models[part] = model
            
        return vq_models
    
    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """Create a single RVQVAE model with specified configuration."""
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        
      
        # Load pretrained weights
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path)['net'])
        return model
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        if 'wavlm' in dict_data:
            wavlm = dict_data["wavlm"].to(self.rank)
        else:
            wavlm = None
        if 'audio_name' in dict_data:
            audio_name = dict_data["audio_name"]
        else:
            audio_name = None
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        ####################
        tar_pose_face = tar_pose[:, :, self.joint_mask_face.astype(bool)]
        tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, 3, 3))
        tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, 3*6)
        #############################
        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg
        
        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
            tar_pose_face =  (tar_pose_face - self.mean_face) / self.std_face
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)
            
            
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        if self.cfg.model.use_face_pose and self.cfg.model.use_exp:
            """
            latent_upper_top: torch.Size([128, 32, 128]), latent_hands_top: torch.Size([128, 32, 128]), latent_lower_top: torch.Size([128, 32, 128]), latent_face_top: torch.Size([128, 32, 128])
            latent_in: torch.Size([128, 32, 512])"""
            
            latent_face_top = self.vq_model_face.map2latent(tar_pose_face)
            print(f"latent_upper_top: {latent_upper_top.shape}, latent_hands_top: {latent_hands_top.shape}, latent_lower_top: {latent_lower_top.shape}, latent_face_top: {latent_face_top.shape}")      
            latent_exp_top = self.vq_model_exp.map2latent(tar_exps)
            
            latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top,latent_face_top,latent_exp_top ], dim=2)/self.args.vqvae_latent_scale
            print(f"latent_in: {latent_in.shape}")          
        
        elif self.cfg.model.use_face_pose:
            latent_face_top = self.vq_model_face.map2latent(tar_pose_face)
            latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top,latent_face_top], dim=2)/self.args.vqvae_latent_scale
        elif self.cfg.model.use_exp:
            latent_exp_top = self.vq_model_exp.map2latent(tar_exps)
            latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top,latent_exp_top], dim=2)/self.args.vqvae_latent_scale
        else:
            latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        style_feature = None
        
        return {
            "in_audio": in_audio,
            "wavlm": wavlm,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "tar_contact": tar_contact,
            "style_feature":style_feature,
            "audio_name": audio_name
        }
        
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
            
        cond_ = {'y':{}}
        cond_['y']['audio'] = loaded_data['in_audio']
        cond_['y']['wavlm'] = loaded_data['wavlm']
        cond_['y']['word'] = loaded_data['in_word']
        cond_['y']['id'] = loaded_data['tar_id']
        cond_['y']['seed'] = loaded_data['latent_in'][:,:self.args.pre_frames]
        cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length//self.args.vqvae_squeeze_scale]) < 1).cuda()
        cond_['y']['style_feature'] = loaded_data['style_feature']
        x0 = loaded_data['latent_in']
        x0 = x0.permute(0, 2, 1).unsqueeze(2)
        print(f"x0 shape: {x0.shape}")#x0 shape: torch.Size([128, 512, 1, 32])
        if epoch > 100:
            g_loss_final = self.model.module.train_forward(cond_, x0, train_consistency=True)['loss']
        else:
            g_loss_final = self.model.module.train_forward(cond_, x0, train_consistency=False)['loss']
        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

        if mode == 'train':
            return g_loss_final
    
    def train(self, epoch):

        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)

            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: 
                    break
        self.opt_s.step(epoch)
    
    
    
    def _g_test(self, loaded_data):
        
        # 设置模式为测试
        mode = 'test'
        # 获取输入数据的形状
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        # 获取输入数据
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        wavlm = loaded_data["wavlm"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        # 计算剩余的帧数
        remain = n%8
        # 如果剩余的帧数不为0，则进行裁剪
        if remain != 0:
            
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            n = n - remain
            
            
            # 初始化存储结果的列表
        rec_all_face = []
        rec_all_exp = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        # 获取vqvae的squeeze_scale
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        # 计算循环的次数
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        # 计算剩余的帧数
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        # 计算每次循环的帧数
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         

        # 循环处理数据
        for i in range(0, roundt):
            # 获取当前帧的数据
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]

            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            if wavlm is not None:
                wavlm_tmp = wavlm[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]
            else:
                wavlm_tmp = None
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            in_x0_tmp = in_x0[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            
            # 如果是第一帧，则使用in_seed_tmp，否则使用last_sample
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]
            
            # 构建条件
            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['wavlm'] = wavlm_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] = in_seed_tmp
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()
            
            
            # 获取模型输出
            sample = self.model(cond_)['latents']
            sample = sample.squeeze().permute(1,0).unsqueeze(0)

            # 保存last_sample
            last_sample = sample.clone()
            
            # 获取latent
            code_dim = self.vq_model_upper.code_dim
            rec_latent_upper = sample[...,:code_dim]
            rec_latent_hands = sample[...,code_dim:code_dim*2]
            rec_latent_lower = sample[...,code_dim*2:code_dim*3]
            if self.cfg.model.use_face_pose:
                rec_latent_face = sample[...,code_dim*3:code_dim*4]
            if self.cfg.model.use_exp:
                rec_latent_exp = sample[...,code_dim*4:code_dim*5]
            
            # 如果是第一帧，则直接添加，否则添加去掉pre_frames的部分
            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
                if self.cfg.model.use_exp:
                    rec_all_exp.append(rec_latent_exp)
                if self.cfg.model.use_face_pose:
                    rec_all_face.append(rec_latent_face)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])
                if self.cfg.model.use_exp:
                    rec_all_exp.append(rec_latent_exp[:, self.args.pre_frames:])
                if self.cfg.model.use_face_pose:
                    rec_all_face.append(rec_latent_face[:, self.args.pre_frames:])
        # 将结果拼接
        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale
        if self.cfg.model.use_exp:
            rec_all_exp = torch.cat(rec_all_exp, dim=1) * self.vqvae_latent_scale
        if self.cfg.model.use_face_pose:
            rec_all_face = torch.cat(rec_all_face, dim=1) * self.vqvae_latent_scale
            
        # 将latent转换为pose
        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        if self.cfg.model.use_exp:
            rec_exp = self.vq_model_exp.latent2origin(rec_all_exp)[0]
        if self.cfg.model.use_face_pose:
            rec_face = self.vq_model_face.latent2origin(rec_all_face)[0]
        # 如果使用trans，则进行转换
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:]
            rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
            rec_trans = torch.zeros_like(rec_trans_v)
            rec_trans = torch.cumsum(rec_trans_v, dim=-2)
            rec_trans[...,1]=rec_trans_v[...,1]
            rec_lower = rec_lower[...,:-3]
        else:
            bs, T, ps = rec_lower.shape
            # 沿y轴正方向平移1
            rec_trans = torch.zeros(bs, T, 3, device=rec_lower.device, dtype=rec_lower.dtype)
            rec_trans[..., 1] = 1.2
        
        # 如果使用pose_norm，则进行转换
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower
            if self.cfg.model.use_face_pose:
                rec_pose_face = rec_face * self.std_face + self.mean_face if self.cfg.model.use_face_pose else None
        # 裁剪剩余的帧数
        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        # 如果使用exp，则使用rec_face，否则使用tar_exps
        if self.cfg.model.use_exp:
            rec_exps = rec_exp
        else:
            rec_exps = tar_exps
            
            
        # 获取leg的pose
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        # 获取upper的pose
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        # 获取lower的pose
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        # 获取hand的pose
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        # 获取face的pose
        if self.cfg.model.use_face_pose:
            rec_pose_face = rec_pose_face.reshape(bs, n, 3, 6)
            rec_pose_face = rc.rotation_6d_to_matrix(rec_pose_face)
            rec_pose_face = rc.matrix_to_axis_angle(rec_pose_face).reshape(bs*n, 3*3)
            rec_pose_face_recover = self.inverse_selection_tensor(rec_pose_face, self.joint_mask_face, bs*n)
        
        # print(f"rec_pose_upper_recover shape: {rec_pose_upper_recover.shape}, rec_pose_lower_recover shape: {rec_pose_lower_recover.shape}, rec_pose_hands_recover shape: {rec_pose_hands_recover.shape}")
        # print(f"rec_pose_face_recover shape: {rec_pose_face_recover.shape if self.cfg.model.use_exp else 'N/A'}")
        # print(f"rec_face_sample: {rec_pose_face_recover[0, :]}")
        # 获取全部的pose
        if self.cfg.model.use_face_pose:
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover +rec_pose_face_recover
        else:
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
            rec_pose[:, 66:75] = tar_pose.reshape(bs*n, 55*3)[:, 66:75]
        # print(f"rec_pose:{rec_pose[0,:]}")
        # rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

        # 将pose转换为6d
        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        # 返回结果
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }
    
    
    
    def test(self, epoch):
        
        # 检查保存结果的路径是否存在，如果存在则返回0
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        # 如果不存在则创建保存结果的路径
        os.makedirs(results_save_path)
        # 记录开始时间
        start_time = time.time()
        # 总长度
        total_length = 0
        # 测试数据列表
        test_seq_list = self.test_data.selected_file
        # 对齐
        align = 0 
        # 潜在输出
        latent_out = []
        # 潜在原始
        latent_ori = []
        # l2损失
        l2_all = 0 
        # lvel损失
        lvel = 0
        # 将模型设置为评估模式
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        # 不计算梯度
        with torch.no_grad():
            # 遍历测试数据
            for its, batch_data in enumerate(self.test_loader):
                # 加载数据
                loaded_data = self._load_data(batch_data)    
                # 模型输出
                net_out = self._g_test(loaded_data)
                # 目标姿态
                tar_pose = net_out['tar_pose']
                # 重构姿态
                rec_pose = net_out['rec_pose']
                # 目标表情
                tar_exps = net_out['tar_exps']
                # 重构表情
                rec_exps = net_out['rec_exps']
                # 目标beta
                tar_beta = net_out['tar_beta']
                # 重构平移
                rec_trans = net_out['rec_trans']
                # 目标平移
                tar_trans = net_out['tar_trans']
                
                # 批次大小，帧数，关节数
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                # 如果帧率不为1，则进行插值
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                

                # 将矩阵转换为旋转6D
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                # 剩余帧数
                remain = n%self.args.vae_test_len
                # 将姿态映射到潜在空间
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())
                
              
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                # 使用SMPL-X模型生成顶点
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )

                vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=rec_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
                        left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
                        right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
                        return_verts=True, 
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
                    )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3]-tar_pose[:,:3],
                    body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
                    right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
                    return_verts=True, 
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
                )  
                # 获取关节点
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                # joints_tar = vertices_tar["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                # 获取面部顶点
                facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :n]
                facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :n]
                # 计算面部速度损失
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
                # 计算l2损失
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n
                
                # 计算l1损失
                _ = self.l1_calculator.run(joints_rec)
                # 如果存在对齐器，则进行对齐
                if self.alignmenter is not None:
                    # 加载音频
                    in_audio_eval, sr = librosa.load(self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    # 重采样
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.args.audio_sr)
                    # 计算偏移量
                    a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
                    # 加载音频和姿态
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                    # 计算对齐分数
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
                 
                # 获取姿态
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                # 加载GT数据
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                # 保存GT数据
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                # 保存重构数据
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                # 累加总长度
                total_length += n

        # 打印l2损失
        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        # 打印lvel损失
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        # 拼接潜在输出
        latent_out_all = np.concatenate(latent_out, axis=0)
        # 拼接潜在原始
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        # 计算FID分数
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        # 打印FID分数
        logger.info(f"fid score: {fid}")
        # 保存FID分数
        self.test_recording("fid", fid, epoch) 
        
        # 计算对齐分数
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        # 打印对齐分数
        logger.info(f"align score: {align_avg}")
        # 保存对齐分数
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        #data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")