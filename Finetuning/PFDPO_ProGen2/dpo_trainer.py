import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import random
from model import clean_sequence
from utils import compute_multi_objective_reward
from pareto_utils import ParetoOptimizer

class ParetoAwareDPOTrainer:
    def __init__(self, policy_network, config,
                 antibacterial_scorer, activity_scorer, toxicity_scorer):
        """Pareto frontier-based DPO trainer"""
        self.policy = policy_network
        self.config = config
        self.device = config.device
        
        # Scorer functions
        self.antibacterial_scorer = antibacterial_scorer
        self.activity_scorer = activity_scorer
        self.toxicity_scorer = toxicity_scorer
        
        # Pareto optimizer
        self.pareto_optimizer = ParetoOptimizer()
        
        # Optimizer
        self.optimizer = optim.Adam(
            [p for n, p in self.policy.model.named_parameters() if p.requires_grad], 
            lr=config.lr
        )
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.pareto_path, exist_ok=True)
        
        # Statistics collection
        self.pareto_stats = {
            'pareto_front_sizes': [],
            'dominated_sizes': [],
            'selection_ratios': []
        }
    
    def collect_pareto_preference_pairs(self, num_samples=64):
        """Collect preference pair data based on Pareto frontier"""
        # Generate more candidate sequences to ensure sufficient diversity
        candidates, _ = self.policy.generate(
            num_sequences=num_samples * 3, 
            min_length=self.config.min_length,
            max_length=self.config.max_length, 
            temperature=1.2
        )
        
        if len(candidates) < 2:
            print("Warning: Insufficient candidate sequences generated")
            return [], []
        
        # Evaluate multi-objective rewards for each sequence
        scored_candidates = []
        for sequence in candidates:
            combined_reward, original_scores, pareto_scores = compute_multi_objective_reward(
                sequence, 
                self.antibacterial_scorer, 
                self.activity_scorer, 
                self.toxicity_scorer, 
                self.config
            )
            
            scored_candidates.append({
                'sequence': sequence,
                'reward': combined_reward,
                'scores': original_scores,  # (antibacterial, activity, toxicity)
                'pareto_scores': pareto_scores  # (antibacterial, activity, -toxicity)
            })
        
        # Find Pareto frontier using Pareto optimizer
        pareto_front, dominated = self.pareto_optimizer.find_pareto_front(scored_candidates)
        
        # Record statistics
        self.pareto_stats['pareto_front_sizes'].append(len(pareto_front))
        self.pareto_stats['dominated_sizes'].append(len(dominated))
        
        print(f"Pareto front size: {len(pareto_front)}, Dominated sequences: {len(dominated)}")
        
        # Build preference pairs
        preference_pairs = self._build_preference_pairs(pareto_front, dominated, num_samples)
        
        return preference_pairs, scored_candidates
    
    def _build_preference_pairs(self, pareto_front, dominated, num_samples):
        """Build preference pairs"""
        preference_pairs = []
        
        if not pareto_front or not dominated:
            print("Warning: Pareto front or dominated set is empty")
            return preference_pairs
        
        # Strategy 1: Pareto front vs dominated sequences
        pareto_vs_dominated_pairs = min(num_samples // 2, len(pareto_front), len(dominated))
        
        for _ in range(pareto_vs_dominated_pairs):
            better_seq = random.choice(pareto_front)
            worse_seq = random.choice(dominated)
            
            preference_pairs.append({
                'better': better_seq['sequence'],
                'worse': worse_seq['sequence'],
                'better_reward': better_seq['reward'],
                'worse_reward': worse_seq['reward'],
                'better_scores': better_seq['scores'],
                'worse_scores': worse_seq['scores'],
                'pair_type': 'pareto_vs_dominated'
            })
        
        # Strategy 2: Intra-Pareto front comparison based on crowding distance
        # if len(pareto_front) >= 2:
        #     crowding_distances = self.pareto_optimizer.crowding_distance(pareto_front)
            
        #     # Sort by crowding distance
        #     sorted_indices = sorted(range(len(pareto_front)), 
        #                           key=lambda i: crowding_distances[i], reverse=True)
            
        #     pareto_internal_pairs = min(num_samples // 4, len(pareto_front) // 2)
            
        #     for i in range(pareto_internal_pairs):
        #         if i * 2 + 1 < len(sorted_indices):
        #             better_idx = sorted_indices[i * 2]  # Larger crowding distance
        #             worse_idx = sorted_indices[i * 2 + 1]  # Smaller crowding distance
                    
        #             better_seq = pareto_front[better_idx]
        #             worse_seq = pareto_front[worse_idx]
                    
        #             preference_pairs.append({
        #                 'better': better_seq['sequence'],
        #                 'worse': worse_seq['sequence'],
        #                 'better_reward': better_seq['reward'],
        #                 'worse_reward': worse_seq['reward'],
        #                 'better_scores': better_seq['scores'],
        #                 'worse_scores': worse_seq['scores'],
        #                 'pair_type': 'pareto_internal'
        #             })
        
        # Strategy 3: Intra-dominated set comparison based on combined reward
        # if len(dominated) >= 2:
        #     dominated_sorted = sorted(dominated, key=lambda x: x['reward'], reverse=True)
        #     dominated_internal_pairs = min(num_samples // 4, len(dominated) // 2)
            
        #     for i in range(dominated_internal_pairs):
        #         if i * 2 + 1 < len(dominated_sorted):
        #             better_seq = dominated_sorted[i * 2]
        #             worse_seq = dominated_sorted[i * 2 + 1]
                    
        #             preference_pairs.append({
        #                 'better': better_seq['sequence'],
        #                 'worse': worse_seq['sequence'],
        #                 'better_reward': better_seq['reward'],
        #                 'worse_reward': worse_seq['reward'],
        #                 'better_scores': better_seq['scores'],
        #                 'worse_scores': worse_seq['scores'],
        #                 'pair_type': 'dominated_internal'
        #             })
        
        return preference_pairs
    
    def dpo_loss(self, better_logps, worse_logps, reference_better_logps, reference_worse_logps, beta=0.1):
        """Calculate DPO loss"""
        logits = beta * (better_logps - reference_better_logps) - beta * (worse_logps - reference_worse_logps)
        losses = -F.logsigmoid(logits)
        return losses.mean()
    
    def compute_seq_logprob(self, sequence, model):
        """Calculate log probability of a sequence under the model"""
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
        """Train on a batch of preference pairs"""
        if not preference_pairs:
            print("Warning: No preference pair data available")
            return self._empty_stats()
        
        # Count different types of preference pairs
        pair_types = {}
        for pair in preference_pairs:
            pair_type = pair.get('pair_type', 'unknown')
            pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
        
        print(f"Preference pair type distribution: {pair_types}")
        
        # Calculate statistics
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
            'pair_types': pair_types,
        }
        
        # Create reference model
        reference_model = self.policy.create_reference_model()
        
        # Batch training
        batch_size = min(self.config.mini_batch_size, len(preference_pairs))
        batch_indices = random.sample(range(len(preference_pairs)), batch_size)
        batch_pairs = [preference_pairs[i] for i in batch_indices]
        
        better_logps = []
        worse_logps = []
        ref_better_logps = []
        ref_worse_logps = []
        
        for pair in batch_pairs:
            better_seq = pair['better']
            worse_seq = pair['worse']
            
            # Calculate log probabilities under current policy
            better_logp = self.compute_seq_logprob(better_seq, self.policy.model)
            worse_logp = self.compute_seq_logprob(worse_seq, self.policy.model)
            
            # Calculate log probabilities under reference model
            with torch.no_grad():
                ref_better_logp = self.compute_seq_logprob(better_seq, reference_model)
                ref_worse_logp = self.compute_seq_logprob(worse_seq, reference_model)
            
            better_logps.append(better_logp)
            worse_logps.append(worse_logp)
            ref_better_logps.append(ref_better_logp)
            ref_worse_logps.append(ref_worse_logp)
        
        # Convert to tensors
        better_logps = torch.stack(better_logps)
        worse_logps = torch.stack(worse_logps)
        ref_better_logps = torch.stack(ref_better_logps)
        ref_worse_logps = torch.stack(ref_worse_logps)
        
        # Calculate loss
        loss = self.dpo_loss(
            better_logps, 
            worse_logps, 
            ref_better_logps, 
            ref_worse_logps, 
            beta=self.config.beta
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.policy.model.parameters() if p.requires_grad], 
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        stats['loss'] = loss.item()
        return stats
    
    def _empty_stats(self):
        """Return empty statistics"""
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
            'pair_types': {},
        }
    
    def train_epoch(self, epoch=None):
        """Train for one epoch"""
        # Collect Pareto-based preference pairs
        preference_pairs, all_candidates = self.collect_pareto_preference_pairs(
            num_samples=self.config.batch_size
        )
        
        if not preference_pairs:
            print("Warning: No preference pairs collected")
            return self._empty_stats_with_overall()
        
        # Visualize Pareto frontier
        if epoch is not None and epoch % 5 == 0:  # Visualize every 5 epochs
            pareto_viz_path = os.path.join(self.config.pareto_path, f'pareto_front_epoch_{epoch}.png')
            pf_size, dom_size = self.pareto_optimizer.visualize_pareto_front(
                all_candidates, save_path=pareto_viz_path, epoch=epoch
            )
            print(f"Pareto frontier visualization saved to: {pareto_viz_path}")
        
        # Train multiple batches
        stats = None
        for dpo_iter in range(self.config.dpo_epochs):
            batch_stats = self.train_batch(preference_pairs)
            if stats is None:
                stats = batch_stats
            else:
                for key in stats:
                    if key != 'pair_types':
                        stats[key] += batch_stats[key]
        
        # Calculate average values
        if stats:
            for key in stats:
                if key != 'pair_types':
                    stats[key] /= self.config.dpo_epochs
        
        # Add overall statistics
        if all_candidates:
            stats['overall_mean_reward'] = np.mean([c['reward'] for c in all_candidates])
            stats['overall_antibacterial'] = np.mean([c['scores'][0] for c in all_candidates])
            stats['overall_activity'] = np.mean([c['scores'][1] for c in all_candidates])
            stats['overall_toxicity'] = np.mean([c['scores'][2] for c in all_candidates])
            stats['pareto_front_size'] = self.pareto_stats['pareto_front_sizes'][-1] if self.pareto_stats['pareto_front_sizes'] else 0
            stats['dominated_size'] = self.pareto_stats['dominated_sizes'][-1] if self.pareto_stats['dominated_sizes'] else 0
        
        return stats
    
    def _empty_stats_with_overall(self):
        """Return empty statistics with overall metrics"""
        empty_stats = self._empty_stats()
        empty_stats.update({
            'overall_mean_reward': 0,
            'overall_antibacterial': 0,
            'overall_activity': 0,
            'overall_toxicity': 0,
            'pareto_front_size': 0,
            'dominated_size': 0,
        })
        return empty_stats
