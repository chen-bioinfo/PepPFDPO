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

CONTEXT = "<|bos|>"  # Using antimicrobial peptide controllers

GENERATION_PARAMS = {
    "max_length": 50,  
    "do_sample": True,  
    "top_p": 0.9,       
    "temperature": 1.2,   
    "pad_token_id": PAD_TOKEN_ID,  
    "eos_token_id": EOS_TOKEN_ID   
}
def cleaned_sequence(seqs):
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    s = []
    for seq in seqs:
        cleaned = ''.join(char for char in seq.upper() if char in VALID_AMINO_ACIDS)
        s.append(cleaned)
    return s
def main():
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-large", trust_remote_code=True).to(DEVICE)
        
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        "your path/best_model"
    ).to(DEVICE)
    
    input_ids = torch.tensor(
        tokenizer.encode(CONTEXT).ids,
        device=DEVICE
    ).unsqueeze(0)

    clean_sequences = []
    batch_size = 100  # Number of sequences generated each time
    total_sequences = 5000  # The total number of sequences generated
    num_batches = total_sequences // batch_size 

    for batch_index in range(num_batches):
        with torch.no_grad():  
            outputs = model.generate(
                input_ids=input_ids,
                num_return_sequences=batch_size, 
                repetition_penalty=1.2,  
                no_repeat_ngram_size=2, 
                **GENERATION_PARAMS
            )
        # print(outputs)
        
        for output in outputs:
            tokens = output.cpu().numpy().tolist()
            eos_pos = tokens.index(EOS_TOKEN_ID) if EOS_TOKEN_ID in tokens else len(tokens)
            valid_tokens = tokens[1: eos_pos]  
            if not valid_tokens:  
                continue
            seq = tokenizer.decode(valid_tokens).replace(str(PAD_TOKEN_ID), "")
            clean_sequences.append(seq)
        

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Batch {batch_index + 1}/{num_batches} completed.")
    rs = cleaned_sequence(clean_sequences)
    # print(rs)
    
    df = pd.DataFrame({'Sequence': rs})
    df.to_csv('your path/sequence.csv', index=False)

if __name__ == "__main__":
    main()
