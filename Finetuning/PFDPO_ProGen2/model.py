import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import copy

def clean_sequence(seq):
    """清理序列，只保留有效的氨基酸字母"""
    if not seq:
        return "" 
    # 只保留有效的氨基酸字母，转为大写
    return ''.join(char for char in seq.upper() if char in "ACDEFGHIKLMNPQRSTVWY")

# 策略网络 - 使用LoRA微调的ProGen2模型
class PolicyNetwork:
    """封装带有LoRA的ProGen2模型作为策略网络"""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.tokenizer = config.tokenizer
        
        # 加载基础模型
        print("正在加载基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path, 
            trust_remote_code=True
        ).to(self.device)
        
        # 加载已微调的LoRA权重
        if config.lora_model_path:
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                config.lora_model_path
            ).to(self.device)
        
        # 冻结基础模型参数，只优化LoRA层
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 确保LoRA层可以被优化
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        
        print(f"可训练参数总数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def create_reference_model(self):
        """创建参考模型的副本 - 用于DPO训练"""
        # 创建当前模型的深拷贝
        reference_model = copy.deepcopy(self.model)
        
        # 确保参考模型的所有参数都是冻结的
        for param in reference_model.parameters():
            param.requires_grad = False
            
        return reference_model
        
    def generate(self, num_sequences=10, min_length=5, max_length=None, temperature=1.2):
        """生成抗菌肽序列，只使用BOS token作为起始"""
        if max_length is None:
            max_length = self.config.max_length
            
        # 保存当前参数状态
        old_model_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        
        # 只使用BOS token作为起始
        batch_inputs = torch.tensor([[self.config.bos_token_id]] * num_sequences, device=self.device)
        
        # 生成序列
        with torch.no_grad():
            # 生成序列
            generated = self.model.generate(
                inputs=batch_inputs,
                min_length=min_length + 1,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.2,  # 惩罚重复token
                no_repeat_ngram_size=2,  # 避免重复的n-gram
                pad_token_id=self.config.pad_token_id,
                eos_token_id=self.config.eos_token_id,
            )
            
            # 解码生成的序列
            outputs = []
            for gen_ids in generated:
                # 移除特殊token
                sequence = ''
                for token_id in gen_ids.tolist():
                    if token_id not in [self.config.bos_token_id, self.config.eos_token_id, self.config.pad_token_id]:
                        token = self.tokenizer.id_to_token(token_id)
                        if token is not None:
                            sequence += token
                seq = clean_sequence(sequence)
                if len(seq) == 0:
                    continue
                outputs.append(seq)
        
        return outputs, old_model_params
    
    def compute_log_probs(self, sequences, old_model_params=None):
        """计算序列在策略下的对数概率"""
        # 如果提供了旧模型参数，则临时使用这些参数计算旧策略的概率
        if old_model_params:
            current_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
            with torch.no_grad():  # 旧策略计算不需要梯度
                for name, param in self.model.named_parameters():
                    if name in old_model_params and param.requires_grad:
                        param.copy_(old_model_params[name])
        
        log_probs = []
        
        for sequence in sequences:
            # 编码序列
            input_ids = [self.config.bos_token_id]
            for char in sequence:
                token_id = self.tokenizer.token_to_id(char)
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
            if old_model_params:
                with torch.no_grad():  # 旧策略计算不需要梯度
                    outputs = self.model(input_ids=inputs, labels=labels)
                    log_prob = -outputs.loss.item()  # 使用item()获取数值
            else:
                # 这里需要计算梯度，所以不使用torch.no_grad()
                outputs = self.model(input_ids=inputs, labels=labels)
                log_prob = -outputs.loss  # 保留梯度
            
            log_probs.append(log_prob)
        
        # 如果使用了旧参数，恢复当前参数
        if old_model_params:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in current_params and param.requires_grad:
                        param.copy_(current_params[name])
        
        # 如果是计算旧策略概率，返回普通张量；否则返回需要梯度的张量
        if old_model_params:
            return torch.tensor(log_probs, device=self.device)
        else:
            # 如果log_probs包含张量，则使用stack；否则转换为张量
            if log_probs and isinstance(log_probs[0], torch.Tensor):
                return torch.stack(log_probs)
            else:
                return torch.tensor(log_probs, device=self.device, requires_grad=True)

# 序列转嵌入向量函数 - 用于评估
def sequence_to_embedding(sequence, model, tokenizer, device, max_length=64):
    """将序列转换为嵌入向量"""
    # 编码序列，添加BOS token
    input_ids = [tokenizer.token_to_id("<|bos|>")]
    for char in sequence:
        token_id = tokenizer.token_to_id(char)
        if token_id is not None:
            input_ids.append(token_id)
    input_ids.append(tokenizer.token_to_id("<|eos|>"))
    
    # 截断或填充
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.token_to_id("<|pad|>")] * pad_len
    
    # 转换为tensor
    input_ids = torch.tensor([input_ids], device=device)
    
    # 获取隐藏状态
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        # 使用最后一层隐藏状态的平均值作为序列嵌入
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1).squeeze()
    
    return embedding