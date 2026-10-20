"""Training scripts for the nine LEXORA artefacts.

CPU baselines (run in `make bootstrap`): L1 triage (TF-IDF + logistic, temperature-scaled), L8 duplicates
(logistic on cosine + metadata), L9 change risk (LightGBM + TreeSHAP).
GPU recipes (run on a CUDA box with requirements-train.txt): L1 ModernBERT, L2 embedder, L3 reranker,
L4 FIM LoRA, L5 QLoRA SFT + DPO, L6 text-to-query QLoRA, L7 incident QLoRA. Each supports --dry-run,
which builds and validates the training data without loading a model.
"""
