import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import random
from model import clean_sequence
from utils import compute_reward

class DPOTrainer:
    def __init__(self, policy_network, config,
                 antibacterial_scorer, activity_scorer, toxicity_scorer):
        """DPO训练器初始化"""
        self.policy = policy_network
        self.config = config
        self.device = config.device
        
        # 打分器函数
        self.antibacterial_scorer = antibacterial_scorer
        self.activity_scorer = activity_scorer
        self.toxicity_scorer = toxicity_scorer
        
        # 优化器 - 只优化LoRA层
        self.optimizer = optim.Adam(
            [p for n, p in self.policy.model.named_parameters() if p.requires_grad], 
            lr=config.lr
        )
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
    def collect_preference_pairs(self, num_samples=64):
        """收集偏好对数据"""
        # 生成多个候选序列
        candidates, _ = self.policy.generate(
            num_sequences=num_samples * 2, 
            min_length=self.config.min_length,
            max_length=self.config.max_length, 
            temperature=1.2
        )
        
        if len(candidates) < 2:
            print("警告: 生成的候选序列不足")
            return [], []
        
        # 评估每个序列的奖励
        scored_candidates = []
        for sequence in candidates:
            reward, scores = compute_reward(
                sequence, 
                self.antibacterial_scorer, 
                self.activity_scorer, 
                self.toxicity_scorer, 
                self.config
            )
            
            scored_candidates.append({
                'sequence': sequence,
                'reward': reward,
                'scores': scores
            })
        
        # 按奖励排序
        scored_candidates.sort(key=lambda x: x['reward'], reverse=True)
        
        # 构建偏好对 - 每对包含一个高奖励和一个低奖励序列
        preference_pairs = []
        n = len(scored_candidates)
        
        # 确保我们至少有num_samples个偏好对
        num_pairs = min(num_samples, n // 2)
        
        for i in range(num_pairs):
            # 选择排名较高的序列作为"好"的示例
            better_idx = i
            
            # 选择排名较低的序列作为"坏"的示例
            worse_idx = n - i - 1
            
            if better_idx >= worse_idx:
                break
                
            better_sequence = scored_candidates[better_idx]['sequence']
            worse_sequence = scored_candidates[worse_idx]['sequence']
            
            # 确保两个序列是不同的
            if better_sequence != worse_sequence:
                preference_pairs.append({
                    'better': better_sequence,
                    'worse': worse_sequence,
                    'better_reward': scored_candidates[better_idx]['reward'],
                    'worse_reward': scored_candidates[worse_idx]['reward'],
                    'better_scores': scored_candidates[better_idx]['scores'],
                    'worse_scores': scored_candidates[worse_idx]['scores']
                })
        
        return preference_pairs, scored_candidates
    
    def dpo_loss(self, better_logps, worse_logps, reference_better_logps, reference_worse_logps, beta=0.1):
        """计算DPO损失"""
        # 计算log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))
        logits = beta * (better_logps - reference_better_logps) - beta * (worse_logps - reference_worse_logps)
        
        # 应用sigmoid交叉熵损失 - log(sigmoid(logits))
        losses = -F.logsigmoid(logits)
        
        return losses.mean()
    
    def compute_seq_logprob(self, sequence, model):
        """计算序列在模型下的对数概率"""
        # 编码序列
        input_ids = [self.config.bos_token_id]
        for char in sequence:
            token_id = self.policy.tokenizer.token_to_id(char)
            if token_id is not None:
                input_ids.append(token_id)
        input_ids.append(self.config.eos_token_id)
        
        # 截断过长序列
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]
        
        # 构建输入和标签
        inputs = torch.tensor([input_ids[:-1]], device=self.device)
        labels = torch.tensor([input_ids[1:]], device=self.device)
        
        # 计算对数概率
        outputs = model(input_ids=inputs, labels=labels)
        log_prob = -outputs.loss  # 负交叉熵为对数概率
        
        return log_prob
    
    def train_batch(self, preference_pairs):
        """训练一个批次的偏好对"""
        if not preference_pairs:
            print("警告: 没有偏好对数据")
            return {
                'loss': 0,
                'mean_reward_better': 0,
                'mean_reward_worse': 0,
                'antibacterial_better': 0,
                'antibacterial_worse': 0,
                'activity_better': 0,
                'activity_worse': 0,
                'toxicity_better': 0,
                'toxicity_worse': 0,
            }
        
        # 统计信息
        stats = {
            'loss': 0,
            'mean_reward_better': np.mean([p['better_reward'] for p in preference_pairs]),
            'mean_reward_worse': np.mean([p['worse_reward'] for p in preference_pairs]),
            'antibacterial_better': np.mean([p['better_scores'][0] for p in preference_pairs]),
            'antibacterial_worse': np.mean([p['worse_scores'][0] for p in preference_pairs]),
            'activity_better': np.mean([p['better_scores'][1] for p in preference_pairs]),
            'activity_worse': np.mean([p['worse_scores'][1] for p in preference_pairs]),
            'toxicity_better': np.mean([p['better_scores'][2] for p in preference_pairs]),
            'toxicity_worse': np.mean([p['worse_scores'][2] for p in preference_pairs]),
        }
        
        # 创建参考模型(reference model)的副本 - 冻结参数
        reference_model = self.policy.create_reference_model()
        
        # 计算每个偏好对的损失
        total_loss = 0
        batch_size = min(self.config.mini_batch_size, len(preference_pairs))
        
        # 随机抽取批次
        batch_indices = random.sample(range(len(preference_pairs)), batch_size)
        batch_pairs = [preference_pairs[i] for i in batch_indices]
        
        better_logps = []
        worse_logps = []
        ref_better_logps = []
        ref_worse_logps = []
        
        # 计算策略模型的对数概率
        for pair in batch_pairs:
            better_seq = pair['better']
            worse_seq = pair['worse']
            
            # 计算当前策略下的对数概率
            better_logp = self.compute_seq_logprob(better_seq, self.policy.model)
            worse_logp = self.compute_seq_logprob(worse_seq, self.policy.model)
            
            # 计算参考模型下的对数概率 (不需要梯度)
            with torch.no_grad():
                ref_better_logp = self.compute_seq_logprob(better_seq, reference_model)
                ref_worse_logp = self.compute_seq_logprob(worse_seq, reference_model)
            
            better_logps.append(better_logp)
            worse_logps.append(worse_logp)
            ref_better_logps.append(ref_better_logp)
            ref_worse_logps.append(ref_worse_logp)
        
        # 将列表转换为张量
        better_logps = torch.stack(better_logps)
        worse_logps = torch.stack(worse_logps)
        ref_better_logps = torch.stack(ref_better_logps)
        ref_worse_logps = torch.stack(ref_worse_logps)
        
        # 计算DPO损失
        loss = self.dpo_loss(
            better_logps, 
            worse_logps, 
            ref_better_logps, 
            ref_worse_logps, 
            beta=self.config.beta
        )
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.policy.model.parameters() if p.requires_grad], 
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        stats['loss'] = loss.item()
        return stats
    
    def train_epoch(self):
        """训练一个轮次"""
        # 收集偏好对
        preference_pairs, candidates = self.collect_preference_pairs(num_samples=self.config.batch_size)
        
        if not preference_pairs:
            print("警告: 没有收集到偏好对数据")
            return {
                'loss': 0,
                'mean_reward_better': 0,
                'mean_reward_worse': 0,
                'antibacterial_better': 0,
                'antibacterial_worse': 0,
                'activity_better': 0,
                'activity_worse': 0,
                'toxicity_better': 0,
                'toxicity_worse': 0,
                'overall_mean_reward': 0,
                'overall_antibacterial': 0,
                'overall_activity': 0,
                'overall_toxicity': 0,
            }
        
        # 训练多个批次
        stats = None
        for _ in range(self.config.dpo_epochs):
            batch_stats = self.train_batch(preference_pairs)
            if stats is None:
                stats = batch_stats
            else:
                # 累加统计信息
                for key in stats:
                    stats[key] += batch_stats[key]
        
        # 计算平均值
        if stats:
            for key in stats:
                stats[key] /= self.config.dpo_epochs
        
        # 添加所有候选序列的平均指标
        if candidates:
            stats['overall_mean_reward'] = np.mean([c['reward'] for c in candidates])
            stats['overall_antibacterial'] = np.mean([c['scores'][0] for c in candidates])
            stats['overall_activity'] = np.mean([c['scores'][1] for c in candidates])
            stats['overall_toxicity'] = np.mean([c['scores'][2] for c in candidates])
        
        return stats