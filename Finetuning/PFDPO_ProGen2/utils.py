import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
        # Environment settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data path
        self.data_path = "/geniusland/home/wanglijuan/sci_proj/GA_opt/srcdata/finall/non_redundant_sequences.csv"
        
        # Model paths
        self.base_model_path = "hugohrban/progen2-large"
        self.lora_model_path = "/geniusland/home/wanglijuan/sci_proj/GA_opt/new/checkpoints/5225_64/best_model"
        self.tokenizer_path = "/geniusland/home/wanglijuan/sci_proj/GA_opt/new/tokenizer.json"
        
        # Model save paths
        self.output_dir = "/geniusland/home/wanglijuan/sci_proj/GA_opt/new/DPO/dpo_pf"
        self.model_epoch_path = self.output_dir + '/model_epoch'
        self.picture_path = self.output_dir + '/picture'
        self.results_path = self.output_dir + '/results'
        self.pareto_path = self.output_dir + '/pareto'  # New: Pareto frontier visualization path
        
        # Sequence parameters
        self.max_length = 64
        self.min_length = 5
        self.ideal_length = 25
        
        # LoRA parameters
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.target_modules = ["qkv_proj"]
        
        # DPO parameters
        self.dpo_epochs = 3
        self.batch_size = 64
        self.mini_batch_size = 16
        self.lr = 5e-6
        self.max_grad_norm = 0.5
        self.beta = 0.1
        
        # Pareto optimization parameters (New)
        # self.use_pareto = True  # Whether to use Pareto optimization
        # self.pareto_selection_ratio = 0.7  # Ratio selected from Pareto frontier
        # self.crowding_distance_weight = 0.3  # Weight for crowding distance
        # self.rank_weight = 0.7  # Weight for Pareto rank
        
        # Reward weights (Keep original for backup scoring)
        self.antibacterial_weight = 1.5
        self.activity_weight = 1.5
        self.toxicity_weight = 1.1
        
        # Token IDs
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.bos_token_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|eos|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        self.prompt = '1'

# Multi-objective reward calculation function
def compute_multi_objective_reward(sequence, antibacterial_scorer, activity_scorer, toxicity_scorer, config):
    """
    Calculate multi-objective reward, return scores for each objective and combined reward
    
    Returns:
        tuple: (combined_reward, original_scores, pareto_scores)
    """
    if len(sequence) < 2:
        return 0, (0, 0, 1), (0, 0, -1)  # (combined_reward, original_scores, pareto_scores)
    
    antibacterial_score = antibacterial_scorer(sequence)
    activity_score = activity_scorer(sequence)
    toxicity_score = toxicity_scorer(sequence)
    
    # Original scores
    original_scores = (antibacterial_score, activity_score, toxicity_score)
    
    # Scores for Pareto optimization (note: toxicity is negated)
    pareto_scores = (antibacterial_score, activity_score, -toxicity_score)
    
    # Combined reward (keep original calculation as backup)
    combined_reward = (config.antibacterial_weight * antibacterial_score + 
                      config.activity_weight * activity_score - 
                      config.toxicity_weight * toxicity_score)
    
    return combined_reward, original_scores, pareto_scores

# Keep original function for compatibility
def compute_reward(sequence, antibacterial_scorer, activity_scorer, toxicity_scorer, config):
    combined_reward, original_scores, _ = compute_multi_objective_reward(
        sequence, antibacterial_scorer, activity_scorer, toxicity_scorer, config
    )
    return combined_reward, original_scores

# Sequence processing function
def decode_sequence(token_ids, tokenizer):
    """Convert token IDs back to sequence"""
    tokens = []
    for token_id in token_ids:
        if token_id not in [tokenizer.token_to_id("<|bos|>"), 
                          tokenizer.token_to_id("<|eos|>"), 
                          tokenizer.token_to_id("<|pad|>")]:
            tokens.append(token_id)
    
    return tokenizer.decode(tokens)
