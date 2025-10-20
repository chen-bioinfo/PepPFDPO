import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import copy

def clean_sequence(seq):
    if not seq:
        return "" 
    return ''.join(char for char in seq.upper() if char in "ACDEFGHIKLMNPQRSTVWY")

class PolicyNetwork:
    """Encapsulates the ProGen2 model with LoRA as a policy network"""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.tokenizer = config.tokenizer
        
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path, 
            trust_remote_code=True
        ).to(self.device)
        
        # Loading fine-tuned LoRA weights
        if config.lora_model_path:
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                config.lora_model_path
            ).to(self.device)
        
        # Freeze the basic model parameters and only optimize the LoRA layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Ensure that the LoRA layer can be optimized
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        
        print(f"Total number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def create_reference_model(self):
        """Create a copy of the reference model - for DPO training"""
        # Create a deep copy of the current model
        reference_model = copy.deepcopy(self.model)
        
        # Make sure all parameters of the reference model are frozen
        for param in reference_model.parameters():
            param.requires_grad = False
            
        return reference_model
        
    def generate(self, num_sequences=10, min_length=5, max_length=None, temperature=1.2):
        """Generate antimicrobial peptide sequences, using only BOS token as the starting point"""
        if max_length is None:
            max_length = self.config.max_length
            
        old_model_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        
        batch_inputs = torch.tensor([[self.config.bos_token_id]] * num_sequences, device=self.device)

        with torch.no_grad():
            generated = self.model.generate(
                inputs=batch_inputs,
                min_length=min_length + 1,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.2,  
                no_repeat_ngram_size=2,  
                pad_token_id=self.config.pad_token_id,
                eos_token_id=self.config.eos_token_id,
            )

            outputs = []
            for gen_ids in generated:
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
        """Calculate the logarithmic probability of the sequence under the strategy"""
        # If old model parameters are provided, these are temporarily used to compute the probabilities of the old policy
        if old_model_params:
            current_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
            with torch.no_grad():  
                for name, param in self.model.named_parameters():
                    if name in old_model_params and param.requires_grad:
                        param.copy_(old_model_params[name])
        
        log_probs = []
        
        for sequence in sequences:
            input_ids = [self.config.bos_token_id]
            for char in sequence:
                token_id = self.tokenizer.token_to_id(char)
                if token_id is not None:
                    input_ids.append(token_id)
            input_ids.append(self.config.eos_token_id)
            
            if len(input_ids) > self.config.max_length:
                input_ids = input_ids[:self.config.max_length]
            
            inputs = torch.tensor([input_ids[:-1]], device=self.device)
            labels = torch.tensor([input_ids[1:]], device=self.device)
        
            if old_model_params:
                with torch.no_grad():  
                    outputs = self.model(input_ids=inputs, labels=labels)
                    log_prob = -outputs.loss.item()  
            else:
                outputs = self.model(input_ids=inputs, labels=labels)
                log_prob = -outputs.loss  
            
            log_probs.append(log_prob)
        
        if old_model_params:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in current_params and param.requires_grad:
                        param.copy_(current_params[name])
        
      # If the old policy probability is calculated, a normal tensor is returned; otherwise, a tensor requiring gradient is returned
        if old_model_params:
            return torch.tensor(log_probs, device=self.device)
        else:
            # If log_probs contains a tensor, use stack; otherwise convert to a tensor
            if log_probs and isinstance(log_probs[0], torch.Tensor):
                return torch.stack(log_probs)
            else:
                return torch.tensor(log_probs, device=self.device, requires_grad=True)

def sequence_to_embedding(sequence, model, tokenizer, device, max_length=64):
     """Convert sequence to embedding vector"""
    # Encode sequence and add BOS token
    input_ids = [tokenizer.token_to_id("<|bos|>")]
    for char in sequence:
        token_id = tokenizer.token_to_id(char)
        if token_id is not None:
            input_ids.append(token_id)
    input_ids.append(tokenizer.token_to_id("<|eos|>"))
    
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.token_to_id("<|pad|>")] * pad_len
    
    input_ids = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1).squeeze()
    
    return embedding
