# PepPFDPO
PepPFDPO: Multi-Objective Antimicrobial Peptide Generation using Pareto-Frontier Enhanced Direct Preference Optimization
# Abstract
Antimicrobial peptide (AMP) drug discovery is challenged by simultaneously optimizing conflicting objectives, such as activity and toxicity. While generative protein language models (ProtLMs) offer potential, they often struggle to effectively achieve large-scale AMP data. In this study, we propose PepPFDPO, a novel framework for multi-objective AMP optimization. PepPFDPO fine-tunes a pretrained generative ProtLM (e.g. ProGen2) based on a parameter efficient fine-tuning via Low-Rank Adaptation (LoRA) and a novel Pareto-Frontier enhanced Direct Preference Optimization (PFDPO) strategy. Crucially, PFDPO incorporates the Pareto non-dominated concept into preference learning, enabling the model to generate sequences that optimally balance multiple competing objectives. Experimental results demonstrate that PepPFDPO exhibits superior performance compared to state-of-the-art methods across multiple functional properties and generative quality metrics. Therefore, PepPFDPO provides an effective paradigm for multi-objective optimization in generative models, significantly advancing therapeutic peptide discovery.
# Model Structure
<img width="963" height="618" alt="image" src="https://github.com/user-attachments/assets/ac763c7b-ae17-45d8-9a2e-326fd6540a80" />

# Usage Guide
