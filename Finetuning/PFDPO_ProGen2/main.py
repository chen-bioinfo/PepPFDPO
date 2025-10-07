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

# 初始化预测器 加载打分模型
toxic_predictor = ToxinPred3(threshold=0.5, model=1)
bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/mic_classifier/')
mic_classifier_model = mic_classifier()
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# 设置随机种子以便复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_sequence(seq):
    """清理序列，只保留有效的氨基酸字母"""
    if not seq:
        return "" 
    # 只保留有效的氨基酸字母，转为大写
    return ''.join(char for char in seq.upper() if char in VALID_AMINO_ACIDS)

# 导入打分器函数
def antibacterial_scorer(sequence):
    if len(sequence) < 3:
        return 0
    """抗菌能力预测器"""
    tmp = [sequence]
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
    pred_amp = amp_classifier_model.predict(pad_seq)
    return float(pred_amp[0][0])

def activity_scorer(sequence):
    if len(sequence) < 3:
        return 0
    """抗菌活性预测器"""
    tmp = [sequence]
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
    pred_mic = mic_classifier_model.predict(pad_seq)
    return float(pred_mic[0][0])

def toxicity_scorer(sequence):
    if len(sequence) < 3:
        return 1
    """毒性打分器"""
    r_tox = toxic_predictor.predict_sequence(sequence)
    return float(r_tox['ML Score'])


def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载配置
    config = Config()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_epoch_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.picture_path, exist_ok=True)
    
    # 初始化策略网络 - 使用LoRA微调的ProGen2模型
    policy = PolicyNetwork(config)
    
    # 初始化DPO训练器
    trainer = DPOTrainer(
        policy_network=policy,
        config=config,
        antibacterial_scorer=antibacterial_scorer,
        activity_scorer=activity_scorer,
        toxicity_scorer=toxicity_scorer,
    )
    
    # 训练模型
    print("开始DPO训练...")
    history = []
    best_reward = -float('inf')
    
    num_epochs = 100  # 总训练轮次
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        # 训练一个轮次
        stats = trainer.train_epoch()
        history.append(stats)
        
        # 打印统计信息
        print(f"轮次 {epoch+1}/{num_epochs}")
        print(f"  DPO损失: {stats['loss']:.4f}")
        print(f"  好序列奖励: {stats['mean_reward_better']:.4f}")
        print(f"  差序列奖励: {stats['mean_reward_worse']:.4f}")
        print(f"  好序列抗菌能力: {stats['antibacterial_better']:.4f}")
        print(f"  好序列抗菌活性: {stats['activity_better']:.4f}")
        print(f"  好序列毒性: {stats['toxicity_better']:.4f}")
        print(f"  总体平均奖励: {stats.get('overall_mean_reward', 0):.4f}")
        
        # 生成和评估序列
        generated_sequences, _ = policy.generate(
            num_sequences=100, 
            temperature=1.2  # 可变的温度参数
        )
        
        # 评估生成的序列
        eval_results = evaluate_sequences(
            generated_sequences, 
            antibacterial_scorer, 
            activity_scorer, 
            toxicity_scorer, 
            config
        )
        
        # 保存评估结果
        best_seq = save_evaluation_results(eval_results, epoch+1, config.results_path)
        
        # 如果当前模型性能最好，保存模型
        if best_seq['avg_reward'] > best_reward:
            best_reward = best_seq['avg_reward']
            policy.model.save_pretrained(os.path.join(config.output_dir, f"best_model"))
            print(f'best model at epoch {epoch+1}')
            
            # 保存最佳序列
            with open(os.path.join(config.results_path, "best_sequence.txt"), "w") as f:
                f.write(f"序列: {best_seq['best_result']['sequence']}\n")
                f.write(f"奖励: {best_seq['best_result']['reward']:.4f}\n")
                f.write(f"抗菌能力: {best_seq['best_result']['antibacterial_score']:.4f}\n")
                f.write(f"抗菌活性: {best_seq['best_result']['activity_score']:.4f}\n")
                f.write(f"毒性: {best_seq['best_result']['toxicity_score']:.4f}\n")
        
        # 每5个轮次后保存模型
        if (epoch+1) % 2 == 0:
            policy.model.save_pretrained(os.path.join(config.model_epoch_path, f"model_epoch{epoch+1}"))
    
    # 保存最终模型
    policy.model.save_pretrained(os.path.join(config.output_dir, "final_model"))
    
    # 绘制训练历史
    plot_training_history(history, config.picture_path)
    
    print("训练完成!")
    print(f"最佳模型已保存到: {os.path.join(config.output_dir, 'best_model')}")

if __name__ == "__main__":
    main()