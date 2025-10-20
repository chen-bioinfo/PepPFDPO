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
        """DPO trainer initialization"""
        self.policy = policy_network
        self.config = config
        self.device = config.device
        
        # Scorer function
        self.antibacterial_scorer = antibacterial_scorer
        self.activity_scorer = activity_scorer
        self.toxicity_scorer = toxicity_scorer
        
        # Optimizer - only optimizes the LoRA layer
        self.optimizer = optim.Adam(
            [p for n, p in self.policy.model.named_parameters() if p.requires_grad], 
            lr=config.lr
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
        
    def collect_preference_pairs(self, num_samples=64):
          """Collect preference pair data"""
        # Generate multiple candidate sequences
        candidates, _ = self.policy.generate(
            num_sequences=num_samples * 2, 
            min_length=self.config.min_length,
            max_length=self.config.max_length, 
            temperature=1.2
        )
        
        if len(candidates) < 2:
            print("Warning: Insufficient candidate sequences generated")
            return [], []
        
        # Evaluate the reward for each sequence
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
        
        # Sort by reward
        scored_candidates.sort(key=lambda x: x['reward'], reverse=True)
        
        # Construct preference pairs - each pair contains a high reward and a low reward sequence
        preference_pairs = []
        n = len(scored_candidates)
        
        # Make sure we have at least num_samples preference pairs
        num_pairs = min(num_samples, n // 2)
        
        for i in range(num_pairs):
            # Select higher ranked sequences as "good" examples
            better_idx = i
            
            # Select lower ranked sequences as "bad" examples
            worse_idx = n - i - 1
            
            if better_idx >= worse_idx:
                break
                
            better_sequence = scored_candidates[better_idx]['sequence']
            worse_sequence = scored_candidates[worse_idx]['sequence']
            
            # Ensure that the two sequences are different
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
        """Calculating DPO Losses"""
        # 计算log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))
        logits = beta * (better_logps - reference_better_logps) - beta * (worse_logps - reference_worse_logps)
        
        # Apply sigmoid cross entropy loss - log(sigmoid(logits))
        losses = -F.logsigmoid(logits)
        
        return losses.mean()
    
    def compute_seq_logprob(self, sequence, model):
        """Calculate the log probability of the sequence under the model"""
        input_ids = [self.config.bos_token_id]
        for char in sequence:
            token_id = self.policy.tokenizer.token_to_id(char)
            if token_id is not None:
                input_ids.append(token_id)
        input_ids.append(self.config.eos_token_id)
        
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]
        
        inputs = torch.tensor([input_ids[:-1]], device=self.device)
        labels = torch.tensor([input_ids[1:]], device=self.device)

        outputs = model(input_ids=inputs, labels=labels)
        log_prob = -outputs.loss  
        
        return log_prob
    
    def train_batch(self, preference_pairs):
        """Training a batch of preference pairs"""
        if not preference_pairs:
            print("Warning: No preference for data")
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
        
        # Create a copy of the reference model - freeze parameters
        reference_model = self.policy.create_reference_model()
        
        # Calculate the loss for each preference pair
        total_loss = 0
        batch_size = min(self.config.mini_batch_size, len(preference_pairs))
        
        batch_indices = random.sample(range(len(preference_pairs)), batch_size)
        batch_pairs = [preference_pairs[i] for i in batch_indices]
        
        better_logps = []
        worse_logps = []
        ref_better_logps = []
        ref_worse_logps = []
        
        # Calculate the log probability of the strategy model
        for pair in batch_pairs:
            better_seq = pair['better']
            worse_seq = pair['worse']
            
            # Calculate the logarithmic probability under the current strategy
            better_logp = self.compute_seq_logprob(better_seq, self.policy.model)
            worse_logp = self.compute_seq_logprob(worse_seq, self.policy.model)
            
            # Calculate the log probability under the reference model (no gradient required)
            with torch.no_grad():
                ref_better_logp = self.compute_seq_logprob(better_seq, reference_model)
                ref_worse_logp = self.compute_seq_logprob(worse_seq, reference_model)
            
            better_logps.append(better_logp)
            worse_logps.append(worse_logp)
            ref_better_logps.append(ref_better_logp)
            ref_worse_logps.append(ref_worse_logp)
        
        better_logps = torch.stack(better_logps)
        worse_logps = torch.stack(worse_logps)
        ref_better_logps = torch.stack(ref_better_logps)
        ref_worse_logps = torch.stack(ref_worse_logps)

        loss = self.dpo_loss(
            better_logps, 
            worse_logps, 
            ref_better_logps, 
            ref_worse_logps, 
            beta=self.config.beta
        )

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
        """Train for one round"""
        # Collect preference pairs
        preference_pairs, candidates = self.collect_preference_pairs(num_samples=self.config.batch_size)
        
        if not preference_pairs:
            print("Warning: No preference data collected")
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
        
        # Train multiple batches
        stats = None
        for _ in range(self.config.dpo_epochs):
            batch_stats = self.train_batch(preference_pairs)
            if stats is None:
                stats = batch_stats
            else:
                # Accumulated statistics
                for key in stats:
                    stats[key] += batch_stats[key]
        
        if stats:
            for key in stats:
                stats[key] /= self.config.dpo_epochs
        
        if candidates:
            stats['overall_mean_reward'] = np.mean([c['reward'] for c in candidates])
            stats['overall_antibacterial'] = np.mean([c['scores'][0] for c in candidates])
            stats['overall_activity'] = np.mean([c['scores'][1] for c in candidates])
            stats['overall_toxicity'] = np.mean([c['scores'][2] for c in candidates])
        
        return stats
