import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from utils import compute_reward

def evaluate_sequences(sequences, antibacterial_scorer, activity_scorer, toxicity_scorer, config):
    """评估生成的序列"""
    results = []
    
    for sequence in sequences:
        reward, scores = compute_reward(
            sequence, 
            antibacterial_scorer, 
            activity_scorer, 
            toxicity_scorer, 
            config
        )
        
        results.append({
            'sequence': sequence,
            'reward': reward,
            'antibacterial_score': scores[0],
            'activity_score': scores[1],
            'toxicity_score': scores[2],
        })
    
    return results

def plot_training_history(history, output_dir):
    """绘制训练历史图表"""
    epochs = range(1, len(history) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [h['loss'] for h in history], label='DPO_Loss')
    plt.title('DPO_Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    
    # 奖励曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [h['mean_reward_better'] for h in history], label='reward_better')
    plt.plot(epochs, [h['mean_reward_worse'] for h in history], label='reward_worse')
    plt.plot(epochs, [h.get('overall_mean_reward', 0) for h in history], label='overall_reward')
    plt.title('mean_reward')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.legend()
    
    # 抗菌能力分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [h['antibacterial_better'] for h in history], label='better-AMP')
    plt.plot(epochs, [h['antibacterial_worse'] for h in history], label='worse-AMP')
    plt.plot(epochs, [h['activity_better'] for h in history], label='better-MIC')
    plt.plot(epochs, [h['activity_worse'] for h in history], label='worse-MIC')
    plt.title('Antibacterial index')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    
    # 毒性曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [h['toxicity_better'] for h in history], label='better-Toxic')
    plt.plot(epochs, [h['toxicity_worse'] for h in history], label='worse-Toxic')
    plt.title('Toxic-score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def save_evaluation_results(results, epoch, output_dir):
    """保存评估结果到文件"""
    output_file = os.path.join(output_dir, f'evaluation_epoch_{epoch}.txt')
    
    # 按奖励降序排序
    sorted_results = sorted(results, key=lambda x: x['reward'], reverse=True)
    
    with open(output_file, 'w') as f:
        f.write(f"{'序列':<50} {'奖励':<10} {'抗菌能力':<10} {'抗菌活性':<10} {'毒性':<10}\n")
        f.write("-" * 100 + "\n")
        
        for result in sorted_results:
            f.write(f"{result['sequence']:<50} {result['reward']:<10.4f} "
                    f"{result['antibacterial_score']:<10.4f} {result['activity_score']:<10.4f} "
                    f"{result['toxicity_score']:<10.4f}\n")
            
    print(f"评估结果已保存到 {output_file}")
    # 计算reward的平均值
    avg_reward = sum(r['reward'] for r in results) / len(results) if results else 0.0
    # 返回最佳序列
    return {
        'avg_reward': avg_reward,
        'best_result': sorted_results[0] if sorted_results else None
    }