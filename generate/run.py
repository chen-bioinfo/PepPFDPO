import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import pandas as pd
from torch.utils.data import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import Tokenizer
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file("your path/tokenizer.json")
BOS_TOKEN_ID = tokenizer.token_to_id("<|bos|>") 
EOS_TOKEN_ID = tokenizer.token_to_id("<|eos|>")
PAD_TOKEN_ID = tokenizer.token_to_id("<|pad|>") 

CONTEXT = "<|bos|>"  # 使用抗菌肽控制符

GENERATION_PARAMS = {
    "max_length": 50,  # 最长生成长度
    "do_sample": True,  # 是否使用采样生成
    "top_p": 0.9,       # nucleus sampling 参数
    "temperature": 1.2,   # 温度参数
    "pad_token_id": PAD_TOKEN_ID,  # 填充 token ID
    "eos_token_id": EOS_TOKEN_ID   # 结束 token ID
}
def cleaned_sequence(seqs):
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    s = []
    for seq in seqs:
        cleaned = ''.join(char for char in seq.upper() if char in VALID_AMINO_ACIDS)
        s.append(cleaned)
    return s
def main():
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large", trust_remote_code=True).to(DEVICE)
        
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(
        base_model,
        "your path/best_model"
    ).to(DEVICE)
    
    # 编码输入
    input_ids = torch.tensor(
        tokenizer.encode(CONTEXT).ids,
        device=DEVICE
    ).unsqueeze(0)

    clean_sequences = []
    batch_size = 100  # 每次生成的序列数量
    total_sequences = 5000  # 总共生成的序列数量
    num_batches = total_sequences // batch_size  # 计算批次

    for batch_index in range(num_batches):
        with torch.no_grad():  # 禁用梯度计算
            outputs = model.generate(
                input_ids=input_ids,
                num_return_sequences=batch_size,  # 当前批次生成数量
                repetition_penalty=1.2,  # 惩罚重复token
                no_repeat_ngram_size=2,  # 避免重复的n-gram
                **GENERATION_PARAMS
            )
        # print(outputs)
        
        for output in outputs:
            tokens = output.cpu().numpy().tolist()
            # 查找 EOS 的位置，若不存在则取全部
            eos_pos = tokens.index(EOS_TOKEN_ID) if EOS_TOKEN_ID in tokens else len(tokens)
            # 移除 BOS 和 EOS，保留中间 token
            valid_tokens = tokens[1: eos_pos]  # 去除 BOS
            if not valid_tokens:  # 跳过空序列
                continue
            # 转换为字符串并移除 PAD
            seq = tokenizer.decode(valid_tokens).replace(str(PAD_TOKEN_ID), "")
            clean_sequences.append(seq)
        
        # 清理显存
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Batch {batch_index + 1}/{num_batches} completed.")
    rs = cleaned_sequence(clean_sequences)
    # print(rs)
    # 保存生成的序列到 CSV 文件
    df = pd.DataFrame({'Sequence': rs})
    df.to_csv('your path/sequence.csv', index=False)

if __name__ == "__main__":
    main()
