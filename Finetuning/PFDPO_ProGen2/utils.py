import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        # 环境设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据路径
        self.data_path = "/geniusland/home/wanglijuan/sci_proj/GA_opt/srcdata/finall/non_redundant_sequences.csv"
        
        # 模型路径
        self.base_model_path = "hugohrban/progen2-large"
        self.lora_model_path = "/geniusland/home/wanglijuan/sci_proj/GA_opt/new/checkpoints/5225_64/best_model"
        self.tokenizer_path = "/geniusland/home/wanglijuan/sci_proj/GA_opt/new/tokenizer.json"
        
        # 模型保存路径
        self.output_dir = "/geniusland/home/wanglijuan/sci_proj/GA_opt/new/DPO/dpo_1.5_mutiple"
        self.model_epoch_path = self.output_dir + '/model_epoch'
        self.picture_path = self.output_dir + '/picture'
        self.results_path = self.output_dir + '/results'
        
        # 序列参数
        self.max_length = 64  # 与LoRA训练保持一致
        self.min_length = 5  # 最小序列长度 
        self.ideal_length = 25  # 理想长度设为25
        
        # LoRA参数
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.target_modules = ["qkv_proj"]
        
        # DPO参数
        self.dpo_epochs = 3  # DPO更新轮次
        self.batch_size = 32  # 每批训练数据大小
        self.mini_batch_size = 8  # DPO小批量大小
        self.lr = 5e-6  # 学习率
        self.max_grad_norm = 0.5  # 梯度裁剪
        self.beta = 0.1  # DPO温度参数
        
        # 奖励权重
        self.antibacterial_weight = 1.5  # 抗菌能力权重
        self.activity_weight = 1.5  # 抗菌活性权重
        self.toxicity_weight = 1.1  # 毒性权重(负面影响)
        # self.diversity_weight = 0.4  # 多样性奖励的权重
        # self.max_repeat_ratio = 0.3  # 单个氨基酸的最大占比
        
        # Token IDs
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.bos_token_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|eos|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        self.prompt = '1'


# 计算奖励函数
def compute_reward(sequence, antibacterial_scorer, activity_scorer, toxicity_scorer, config):
    if len(sequence) < 2:
        return 0, (0, 0, 1)
    """
    Returns:
        float: 综合奖励分数, tuple: 各个分数
    """
    antibacterial_score = antibacterial_scorer(sequence)
    activity_score = activity_scorer(sequence)
    toxicity_score = toxicity_scorer(sequence)
    
    # 奖励计算: 高抗菌能力和活性，低毒性
    # base_reward = (config.antibacterial_weight * antibacterial_score + 
    #           config.activity_weight * activity_score - 
    #           config.toxicity_weight * toxicity_score)
    base_reward = antibacterial_score * activity_score * (1 - toxicity_score)
    # 多样性奖励
    # diversity_reward = (0.3 * diversity_score - 
    #                     0.4 * repeat_penalty - 
    #                     0.2 * ngram_penalty)
    
   
    # reward = base_reward + config.diversity_weight * diversity_reward + length_reward
    # 奖励计算: 高抗菌能力和活性，低毒性
    
    return base_reward, (antibacterial_score, activity_score, toxicity_score)

# 序列处理函数
def decode_sequence(token_ids, tokenizer):
    """将token ID转换回序列"""
    # 移除特殊token（BOS, EOS, PAD）
    tokens = []
    for token_id in token_ids:
        if token_id not in [tokenizer.token_to_id("<|bos|>"), 
                          tokenizer.token_to_id("<|eos|>"), 
                          tokenizer.token_to_id("<|pad|>")]:
            tokens.append(token_id)
    
    return tokenizer.decode(tokens)