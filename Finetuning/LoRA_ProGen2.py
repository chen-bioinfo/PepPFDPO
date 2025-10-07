import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
# from models.progen.modeling_progen import ProGenForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EvalPrediction, DataCollatorForLanguageModeling, DataCollatorWithPadding
from tokenizers import Tokenizer
import random

# 配置参数
DATA_PATH = "your path/non_redundant_sequences.csv"
OUTPUT_PATH = "your path/"
tokenizer = Tokenizer.from_file("your path/tokenizer.json")
BOS_TOKEN_ID = tokenizer.token_to_id("<|bos|>") 
EOS_TOKEN_ID = tokenizer.token_to_id("<|eos|>")
PAD_TOKEN_ID = tokenizer.token_to_id("<|pad|>") 

 # 最大序列长度（含控制符）
MAX_LENGTH = 64       

# LoRA 参数
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 训练参数
TRAIN_PARAMS = {
    "output_dir": OUTPUT_PATH,
    "num_train_epochs": 50,
    "per_device_train_batch_size": 32,
    'per_device_eval_batch_size': 32,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    'logging_dir':'your path/log', 
    'save_total_limit':5, 
    'load_best_model_at_end':True,
    "optim": "adamw_torch_fused",  
}

class AMPDataset(Dataset):
    def __init__(self, data, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.sequences = []

        for s in data:
            encoded = tokenizer.encode(s).ids
            self.sequences.append(encoded)
        # with open(DATA_PATH) as f:
        #     for line in f:
        #         seq = line.strip()
        #         if not seq:
        #             continue  
        #         encoded = tokenizer.encode(seq).ids
        #         self.sequences.append(encoded)

    def __len__(self):
        return len(self.sequences)
                
    def __getitem__(self, idx):

        seq = [BOS_TOKEN_ID] + self.sequences[idx]
        if len(seq) >= MAX_LENGTH:
            seq = seq[0 : MAX_LENGTH-2]
        pad_len = MAX_LENGTH - len(seq) - 1
        input_ids = seq + [EOS_TOKEN_ID] + [PAD_TOKEN_ID] * pad_len
        labels = input_ids[1:] + [PAD_TOKEN_ID]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }

def main():
    # 加载分词器
    # tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer = Tokenizer.from_file("your path/tokenizer.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large", trust_remote_code=True).to(device)
    # 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv_proj"],
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    data = pd.read_csv(DATA_PATH)
    sequences = data['Sequence']
    random.shuffle(sequences)

    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    eval_sequences = sequences[train_size:]    
    # 打印可训练参数
    # model.print_trainable_parameters()
    
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAIN_PARAMS),
        train_dataset=AMPDataset(train_sequences, tokenizer),
        eval_dataset=AMPDataset(eval_sequences, tokenizer)
    )
    
    # 开始训练
    trainer.train()
    # 评估模型
    eval_results = trainer.evaluate()
    # print("Evaluation results:", eval_results)
    model.save_pretrained(OUTPUT_PATH)
    trainer.save_model("/your path/best_model")
    

if __name__ == "__main__":
    main()