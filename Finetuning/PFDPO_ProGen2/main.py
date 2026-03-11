import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import torch
import numpy as np
import math
import sys
import random
from tqdm import tqdm
from utils import Config
from model import PolicyNetwork
from dpo_trainer import ParetoAwareDPOTrainer  # Changed to use Pareto-aware trainer
from evaluation import evaluate_sequences, plot_training_history, save_evaluation_results
from pareto_utils import ParetoOptimizer  # New addition
from amp.utils import basic_model_serializer
import amp.data_utils.sequence as du_sequence
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from toxinpred3.toxic import ToxinPred3

# Initialize predictors
toxic_predictor = ToxinPred3(threshold=0.5, model=1)
bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/mic_classifier/')
mic_classifier_model = mic_classifier()
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_sequence(seq):
    if not seq:
        return "" 
    return ''.join(char for char in seq.upper() if char in VALID_AMINO_ACIDS)

def antibacterial_scorer(sequence):
    if len(sequence) < 3:
        return 0
    tmp = [sequence]
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
    pred_amp = amp_classifier_model.predict(pad_seq)
    return float(pred_amp[0][0])

def activity_scorer(sequence):
    if len(sequence) < 3:
        return 0
    tmp = [sequence]
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
    pred_mic = mic_classifier_model.predict(pad_seq)
    return float(pred_mic[0][0])

def toxicity_scorer(sequence):
    if len(sequence) < 3:
        return 1
    r_tox = toxic_predictor.predict_sequence(sequence)
    return float(r_tox['ML Score'])

def main():
    set_seed(42)
    
    config = Config()
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_epoch_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.picture_path, exist_ok=True)
    os.makedirs(config.pareto_path, exist_ok=True)  # New addition
    
    # Initialize policy network
    policy = PolicyNetwork(config)
    
    # Initialize Pareto-aware DPO trainer
    trainer = ParetoAwareDPOTrainer(
        policy_network=policy,
        config=config,
        antibacterial_scorer=antibacterial_scorer,
        activity_scorer=activity_scorer,
        toxicity_scorer=toxicity_scorer,
    )
    
    print("Starting Pareto frontier-based DPO training...")
    history = []
    best_reward = -float('inf')
    
    num_epochs = 100
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Train for one epoch
        stats = trainer.train_epoch(epoch=epoch)
        history.append(stats)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  DPO Loss: {stats['loss']:.4f}")
        print(f"  Better Sequence Reward: {stats['mean_reward_better']:.4f}")
        print(f"  Worse Sequence Reward: {stats['mean_reward_worse']:.4f}")
        print(f"  Pareto Front Size: {stats.get('pareto_front_size', 0)}")
        print(f"  Number of Dominated Sequences: {stats.get('dominated_size', 0)}")
        print(f"  Preference Pair Types: {stats.get('pair_types', {})}")
        print(f"  Better Sequence Antibacterial Activity: {stats['antibacterial_better']:.4f}")
        print(f"  Better Sequence Antimicrobial Activity: {stats['activity_better']:.4f}")
        print(f"  Better Sequence Toxicity: {stats['toxicity_better']:.4f}")
        print(f"  Overall Average Reward: {stats.get('overall_mean_reward', 0):.4f}")
        
        # Generate and evaluate sequences
        generated_sequences, _ = policy.generate(
            num_sequences=100, 
            temperature=1.2
        )
        
        # Evaluate generated sequences
        eval_results = evaluate_sequences(
            generated_sequences, 
            antibacterial_scorer, 
            activity_scorer, 
            toxicity_scorer, 
            config
        )
        
        # Save evaluation results
        best_seq = save_evaluation_results(eval_results, epoch+1, config.results_path)
        
        # Save best model
        if best_seq['avg_reward'] > best_reward:
            best_reward = best_seq['avg_reward']
            policy.model.save_pretrained(os.path.join(config.output_dir, f"best_model"))
            print(f'Best model saved at epoch {epoch+1}')
            
            # Save best sequence
            with open(os.path.join(config.results_path, "best_sequence.txt"), "w") as f:
                f.write(f"Sequence: {best_seq['best_result']['sequence']}\n")
                f.write(f"Reward: {best_seq['best_result']['reward']:.4f}\n")
                f.write(f"Antibacterial Score: {best_seq['best_result']['antibacterial_score']:.4f}\n")
                f.write(f"Activity Score: {best_seq['best_result']['activity_score']:.4f}\n")
                f.write(f"Toxicity Score: {best_seq['best_result']['toxicity_score']:.4f}\n")
        
        # Save model every 2 epochs
        if (epoch+1) % 2 == 0:
            policy.model.save_pretrained(os.path.join(config.model_epoch_path, f"model_epoch{epoch+1}"))
    
    # Save final model
    policy.model.save_pretrained(os.path.join(config.output_dir, "final_model"))
    
    # Plot training history (including Pareto-related visualizations)
    plot_training_history(history, config.picture_path)
    plot_pareto_statistics(history, config.picture_path)  # New Pareto statistics plot
    
    print("Training completed!")
    print(f"Best model saved to: {os.path.join(config.output_dir, 'best_model')}")
    print(f"Pareto frontier visualizations saved to: {config.pareto_path}")

def plot_pareto_statistics(history, output_dir):
    """Plot Pareto-related statistical charts"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Pareto front size trend
    plt.subplot(1, 3, 1)
    pareto_sizes = [h.get('pareto_front_size', 0) for h in history]
    plt.plot(epochs, pareto_sizes, 'b-', label='Pareto Front Size')
    plt.title('Pareto Front Size Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dominated sequences count trend
    plt.subplot(1, 3, 2)
    dominated_sizes = [h.get('dominated_size', 0) for h in history]
    plt.plot(epochs, dominated_sizes, 'r-', label='Dominated Size')
    plt.title('Dominated Sequences Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Pareto front ratio
    plt.subplot(1, 3, 3)
    total_candidates = [p + d for p, d in zip(pareto_sizes, dominated_sizes)]
    pareto_ratios = [p / t if t > 0 else 0 for p, t in zip(pareto_sizes, total_candidates)]
    plt.plot(epochs, pareto_ratios, 'g-', label='Pareto Ratio')
    plt.title('Pareto Front Ratio Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
