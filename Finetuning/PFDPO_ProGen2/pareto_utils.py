import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ParetoOptimizer:
    def __init__(self, objectives=['antibacterial', 'activity', 'toxicity']):
        """
        Pareto optimization class for multi-objective sequence evaluation
        
        Args:
            objectives: List of objective names. Note: toxicity will be automatically negated 
                       (since we aim to minimize toxicity)
        """
        self.objectives = objectives
        self.pareto_history = []  # Record Pareto frontiers from each iteration
    
    def is_pareto_dominated(self, point1, point2):
        """
        Check if point1 is Pareto-dominated by point2
        
        Args:
            point1, point2: tuple of (antibacterial, activity, -toxicity)
        Returns:
            bool: True if point1 is dominated by point2
        """
        # Point2 is better or equal in all dimensions, and strictly better in at least one dimension
        better_or_equal = all(p2 >= p1 for p1, p2 in zip(point1, point2))
        strictly_better = any(p2 > p1 for p1, p2 in zip(point1, point2))
        return better_or_equal and strictly_better
    
    def find_pareto_front(self, candidates):
        """
        Identify the Pareto frontier from candidate sequences
        
        Args:
            candidates: list of dicts with 'sequence', 'scores', 'reward'
        Returns:
            pareto_front: list of non-dominated candidates
            dominated: list of dominated candidates
        """
        pareto_front = []
        dominated = []
        
        # Create objective points for each candidate (antibacterial, activity, -toxicity)
        points = []
        for candidate in candidates:
            # Note: Toxicity is negated because we aim to minimize it
            point = (
                candidate['scores'][0],  # antibacterial score
                candidate['scores'][1],  # activity score  
                -candidate['scores'][2]  # negative toxicity score
            )
            points.append(point)
        
        # Check if each point is dominated by any other point
        for i, candidate in enumerate(candidates):
            point_i = points[i]
            is_dominated = False
            
            for j, other_candidate in enumerate(candidates):
                if i != j:
                    point_j = points[j]
                    if self.is_pareto_dominated(point_i, point_j):
                        is_dominated = True
                        break
            
            if is_dominated:
                dominated.append(candidate)
            else:
                pareto_front.append(candidate)
        
        return pareto_front, dominated
    
    def compute_pareto_rank(self, candidates):
        """
        Calculate Pareto rank for each candidate sequence
        
        Returns:
            ranks: list of integers, where 0 is the highest rank (Pareto frontier)
        """
        remaining = candidates.copy()
        ranks = {}
        current_rank = 0
        
        while remaining:
            # Find Pareto frontier in current remaining candidates
            current_front, dominated = self.find_pareto_front(remaining)
            
            # Assign ranks
            for candidate in current_front:
                ranks[id(candidate)] = current_rank
            
            # Remove processed candidates
            remaining = dominated
            current_rank += 1
        
        # Return ranked list
        rank_list = [ranks[id(candidate)] for candidate in candidates]
        return rank_list
    
    def crowding_distance(self, front):
        """
        Calculate crowding distance for selection within the same Pareto rank
        
        Args:
            front: List of candidates in the same Pareto frontier
        Returns:
            distances: List of crowding distance values for each candidate
        """
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        n_objectives = 3  # antibacterial, activity, -toxicity
        
        for obj_idx in range(n_objectives):
            # Sort by current objective
            if obj_idx == 0:  # antibacterial
                sorted_indices = sorted(range(len(front)), 
                                      key=lambda i: front[i]['scores'][0], reverse=True)
            elif obj_idx == 1:  # activity
                sorted_indices = sorted(range(len(front)), 
                                      key=lambda i: front[i]['scores'][1], reverse=True)
            else:  # -toxicity
                sorted_indices = sorted(range(len(front)), 
                                      key=lambda i: -front[i]['scores'][2], reverse=True)
            
            # Set boundary points to infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate objective value range
            if obj_idx == 0:
                obj_min = front[sorted_indices[-1]]['scores'][0]
                obj_max = front[sorted_indices[0]]['scores'][0]
            elif obj_idx == 1:
                obj_min = front[sorted_indices[-1]]['scores'][1]
                obj_max = front[sorted_indices[0]]['scores'][1]
            else:
                obj_min = -front[sorted_indices[0]]['scores'][2]
                obj_max = -front[sorted_indices[-1]]['scores'][2]
            
            obj_range = obj_max - obj_min
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for middle points
            for i in range(1, len(sorted_indices) - 1):
                if distances[sorted_indices[i]] != float('inf'):
                    if obj_idx == 0:
                        prev_obj = front[sorted_indices[i-1]]['scores'][0]
                        next_obj = front[sorted_indices[i+1]]['scores'][0]
                    elif obj_idx == 1:
                        prev_obj = front[sorted_indices[i-1]]['scores'][1]
                        next_obj = front[sorted_indices[i+1]]['scores'][1]
                    else:
                        prev_obj = -front[sorted_indices[i-1]]['scores'][2]
                        next_obj = -front[sorted_indices[i+1]]['scores'][2]
                    
                    distances[sorted_indices[i]] += abs(prev_obj - next_obj) / obj_range
        
        return distances
    
    def visualize_pareto_front(self, candidates, save_path=None, epoch=None):
        """
        Visualize Pareto frontier (3D scatter plot)
        
        Args:
            candidates: List of candidate sequences with scores
            save_path: Path to save the visualization (optional)
            epoch: Training epoch number for title (optional)
        
        Returns:
            tuple: (pareto_front_size, dominated_size)
        """
        pareto_front, dominated = self.find_pareto_front(candidates)
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot dominated points
        if dominated:
            dom_antibacterial = [c['scores'][0] for c in dominated]
            dom_activity = [c['scores'][1] for c in dominated]
            dom_toxicity = [c['scores'][2] for c in dominated]
            ax1.scatter(dom_antibacterial, dom_activity, dom_toxicity, 
                       c='red', alpha=0.6, s=20, label='Dominated')
        
        # Plot Pareto frontier points
        if pareto_front:
            pf_antibacterial = [c['scores'][0] for c in pareto_front]
            pf_activity = [c['scores'][1] for c in pareto_front]
            pf_toxicity = [c['scores'][2] for c in pareto_front]
            ax1.scatter(pf_antibacterial, pf_activity, pf_toxicity, 
                       c='blue', alpha=0.8, s=50, label='Pareto Front')
        
        ax1.set_xlabel('Antibacterial Score')
        ax1.set_ylabel('Activity Score')
        ax1.set_zlabel('Toxicity Score')
        ax1.legend()
        ax1.set_title(f'Pareto Front 3D View (Epoch {epoch})' if epoch else 'Pareto Front 3D View')
        
        # 2D projection (Antibacterial vs Activity)
        ax2 = fig.add_subplot(122)
        
        if dominated:
            ax2.scatter(dom_antibacterial, dom_activity, 
                       c='red', alpha=0.6, s=20, label='Dominated')
        
        if pareto_front:
            ax2.scatter(pf_antibacterial, pf_activity, 
                       c='blue', alpha=0.8, s=50, label='Pareto Front')
        
        ax2.set_xlabel('Antibacterial Score')
        ax2.set_ylabel('Activity Score')
        ax2.legend()
        ax2.set_title('Antibacterial vs Activity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return len(pareto_front), len(dominated)
