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
from dpo_trainer import DPOTrainer
from evaluation import evaluate_sequences, plot_training_history, save_evaluation_results
from amp.utils import basic_model_serializer
import amp.data_utils.sequence as du_sequence
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from toxinpred3.toxic import ToxinPred3

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
    """Clean the sequence to keep only valid amino acid letters"""
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
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_epoch_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.picture_path, exist_ok=True)
    
    # Initializing the policy network - ProGen2 model fine-tuned using LoRA
    policy = PolicyNetwork(config)
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        policy_network=policy,
        config=config,
        antibacterial_scorer=antibacterial_scorer,
        activity_scorer=activity_scorer,
        toxicity_scorer=toxicity_scorer,
    )
    
    print("Start DPO training...")
    history = []
    best_reward = -float('inf')
    
    num_epochs = 100  
    for epoch in tqdm(range(num_epochs), desc="training progress"):
        stats = trainer.train_epoch()
        history.append(stats)
        
        print(f"rounds {epoch+1}/{num_epochs}")
        print(f"  DPO loss: {stats['loss']:.4f}")
        print(f"  Good sequence rewards: {stats['mean_reward_better']:.4f}")
        print(f"  Difference sequence reward: {stats['mean_reward_worse']:.4f}")
        print(f"  Good sequence antibacterial ability: {stats['antibacterial_better']:.4f}")
        print(f"  Good sequence antibacterial activity: {stats['activity_better']:.4f}")
        print(f"  good sequence toxicity: {stats['toxicity_better']:.4f}")
        print(f"  overall average reward: {stats.get('overall_mean_reward', 0):.4f}")
        
        # Generating and evaluating sequences
        generated_sequences, _ = policy.generate(
            num_sequences=100, 
            temperature=1.2 
        )
        
        # Evaluate the generated sequence
        eval_results = evaluate_sequences(
            generated_sequences, 
            antibacterial_scorer, 
            activity_scorer, 
            toxicity_scorer, 
            config
        )
    
        best_seq = save_evaluation_results(eval_results, epoch+1, config.results_path)
        
        # If the current model performs best, save the model
        if best_seq['avg_reward'] > best_reward:
            best_reward = best_seq['avg_reward']
            policy.model.save_pretrained(os.path.join(config.output_dir, f"best_model"))
            print(f'best model at epoch {epoch+1}')
            
            # Save the best sequence
            with open(os.path.join(config.results_path, "best_sequence.txt"), "w") as f:
                f.write(f"Sequence: {best_seq['best_result']['sequence']}\n")
                f.write(f"reward: {best_seq['best_result']['reward']:.4f}\n")
                f.write(f"Antibacterial ability: {best_seq['best_result']['antibacterial_score']:.4f}\n")
                f.write(f"Antibacterial activity: {best_seq['best_result']['activity_score']:.4f}\n")
                f.write(f"toxicity: {best_seq['best_result']['toxicity_score']:.4f}\n")

        if (epoch+1) % 2 == 0:
            policy.model.save_pretrained(os.path.join(config.model_epoch_path, f"model_epoch{epoch+1}"))
    
    policy.model.save_pretrained(os.path.join(config.output_dir, "final_model"))
    
    plot_training_history(history, config.picture_path)
    
    print("Training completed!")
    print(f"The best model has been saved to: {os.path.join(config.output_dir, 'best_model')}")

if __name__ == "__main__":
    main()
