# PepPFDPO
PepPFDPO: Multi-Objective Antimicrobial Peptide Generation using Pareto-Frontier Enhanced Direct Preference Optimization
# Abstract
Antimicrobial peptide (AMP) drug discovery is challenged by simultaneously optimizing conflicting objectives, such as activity and toxicity. While generative protein language models (ProtLMs) offer potential, they often struggle to effectively achieve large-scale AMP data. In this study, we propose PepPFDPO, a novel framework for multi-objective AMP optimization. PepPFDPO fine-tunes a pretrained generative ProtLM (e.g. ProGen2) based on a parameter efficient fine-tuning via Low-Rank Adaptation (LoRA) and a novel Pareto-Frontier enhanced Direct Preference Optimization (PFDPO) strategy. Crucially, PFDPO incorporates the Pareto non-dominated concept into preference learning, enabling the model to generate sequences that optimally balance multiple competing objectives. Experimental results demonstrate that PepPFDPO exhibits superior performance compared to state-of-the-art methods across multiple functional properties and generative quality metrics. Therefore, PepPFDPO provides an effective paradigm for multi-objective optimization in generative models, significantly advancing therapeutic peptide discovery.
# Model Structure
<img width="963" height="618" alt="image" src="https://github.com/user-attachments/assets/ac763c7b-ae17-45d8-9a2e-326fd6540a80" />

# Usage Guide
## 1. Installation
Python (Version 3.8 or higher is recommended)

pip install -r requirements.txt

## 2. How to Run
Step 1: Fine-tune ProGen2 using LoRA_ProGen2.py in the Fituning folder.

Step 2: Use the code in the PFDPO_ProGen2 folder in the Fituning folder to fine-tune the ProGen2 after LoRA fine-tuning, replace the path in the code with your own, and then run the main.py file.

Step3：After training the model, you can run run.py in the generate folder to generate the sequence.

Step4: Use the code in the Metric folder to evaluate the lifecycle sequence and the code in the picture folder to visualize the results.

## 3. File and Directory Description
Fituning: Fine-tuning the protein language model.

generate: Generate sequences using the fine-tuned model.

Metric: Evaluate various metrics of the generated sequence.

picture: Visualize the results.

ProtPLM: Protein language models for fine-tuning.

results: Sequences generated in this experiment and the results.

srcdata: Data used to fine-tune the model.

requirements.txt: The environment required for the experiment depends on the experiment.

## Citation
If you use this code or our model for your publication, please cite the original paper.
