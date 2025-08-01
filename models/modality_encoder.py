import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.layer import BasicBlock
from einops import rearrange
import pickle
import math
from models.wavlm.WavLM import WavLM, WavLMConfig


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=2):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1700, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)


class ModalityEncoder(nn.Module):
    def __init__(self, 
                 data_path, 
                 t_fix_pre, 
                 audio_dim, 
                 audio_in=2,
                 raw_audio=False,
                 latent_dim=256,
                 audio_fps=30,
                 use_exp=False,
                 use_face_pose=False,
                 trainer=None
                 ):
        super().__init__()
        self.trainer = trainer
        self.use_exp = use_exp
        self.use_face_pose = use_face_pose
        if str(self.trainer) == 'shortcut_rvqvae':
            if self.use_exp:
                self.joint_num = 4
            else:
                self.joint_num = 3

        else:
            if self.use_face_pose and self.use_exp:
                self.joint_num = 5
            elif self.use_face_pose or self.use_exp:
                self.joint_num = 4
            else:
                self.joint_num = 3
        print(f"modality_encoder:use_exp={use_exp}, use_face_pose={use_face_pose}")
        self.raw_audio = raw_audio
        self.latent_dim = latent_dim
        self.audio_fps = audio_fps

        self.WavEncoder = WavEncoder(audio_dim, audio_in=audio_in)
        self.text_encoder_body = nn.Linear(300, audio_dim) 

        with open(f"{data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=t_fix_pre)
        word_dim = pre_trained_embedding.shape[1]

        if self.raw_audio:
            # load the pre-trained wavlm model
            # self.load_and_freeze_wavlm()
            self.audio_projection = nn.Linear(1024, audio_dim)
        if self.raw_audio:
            if use_exp and use_face_pose:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*self.joint_num)
            elif use_exp or use_face_pose:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*self.joint_num)
            else:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*self.joint_num)
        else:
            if use_exp and use_face_pose:
                self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim*self.joint_num)
            elif use_exp or use_face_pose:
                self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim*self.joint_num)
            else:
                self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim*self.joint_num)  
    
    def forward(self, audio, word, raw_audio=None, squeeze_scale=4):
        # Initial features extraction - single transpose each
        # [B, T, D] -> [T, B, D]
        audio_feat = self.WavEncoder(audio)
        text_feat = self.text_encoder_body(self.text_pre_encoder_body(word))
        
        if raw_audio is not None and self.raw_audio:
            # Keep the same transpose pattern for consistency
            raw_feat = self.extract_wavlm_feats(raw_audio)
            raw_feat = self.audio_projection(raw_audio)
            
            at_feat = torch.cat([audio_feat, raw_feat, text_feat], dim=2)
        else:
            at_feat = torch.cat([audio_feat, text_feat], dim=2)  # [B, T, D]
        
        at_feat = self.mix_audio_text(at_feat)  # [B, T, D']
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), squeeze_scale)
        at_feat = at_feat.transpose(1, 2) # [B, T/scale, D']
        return at_feat

    @torch.no_grad()
    def load_and_freeze_wavlm(self, wavlm_path='./dataloaders/wavlm/WavLM-Base+.pt'):
        checkpoint = torch.load(wavlm_path)
        self.wavlm_cfg = WavLMConfig(checkpoint['cfg'])
        self.audio_encoder = WavLM(self.wavlm_cfg)
        self.audio_encoder.load_state_dict(checkpoint['model'])
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    

    def extract_wavlm_feats(self, wav_input_16khz):
        assert self.audio_encoder is not None, "Please load the wavlm model first"
        # check the input type
        if isinstance(wav_input_16khz, np.ndarray):
            wav_input_16khz = torch.from_numpy(wav_input_16khz)
        if wav_input_16khz.dim() == 1:
            wav_input_16khz = wav_input_16khz.unsqueeze(0)
        wav_input_16khz = wav_input_16khz.cuda()

        if self.wavlm_cfg.normalize:
            wav_input_16khz = F.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        
        wavlm_feats = self.audio_encoder.extract_features(wav_input_16khz)[0]
        wavlm_feats = wavlm_feats.detach() # (bs, seq_len, dim)
        
        target_size = math.ceil(wavlm_feats.shape[1] / 50 * self.audio_fps)
        wavlm_feats = F.interpolate(
            wavlm_feats.transpose(1, 2),
            size=target_size,
            align_corners=True,
            mode='linear'
        ).transpose(1, 2)
        return wavlm_feats
        
