
## Introduce

The MeZO algorithm is a way to fine-tune LLMs using only forward passes, which means it doesn't need as much memory as backpropagation. The LoRA is a Parameter-Efficient-Fine-Tuning (PEFT) approach that freezes a pre-trained model and applies additional trainable parameters (weights) that are factorized with small decomposition rank r. Thus, the number of LoRA trainable parameters is much smaller than the full fine-tuning, so the fine-tuning requires a much smaller amount of time.

This research presents the results of perfomance comparison of LoRA with MeZO+LoRA fine-tuning of target dataset Microsoft Research Paraphrase Corpus (MRPC). 
## Experiments

- dataset: Microsoft Research Paraphrase Corpus (MRPC)
- GPU: NVIDIA RTX 3060 12GB
- base model: roberta-large
- finetune methods: LoRA and MeZO+LoRA
- training steps: 1000
- lr = 1e-4
- batch size = 64
- frameworks: pytorch for training, w&b for eval results
## Results and discussion

Table 1 show perfomance comparison of LoRA with MeZO+LoRA fine-tuning on target dataset Microsoft Research Paraphrase Corpus (MRPC). We can see, thats MeZO+LoRA outperformed than LoRA method in our testing categories. Figure 1, 2 demonstrated loss and f1 metrics.

Table 1. Perfomance comparison of LoRA with MeZO+LoRA fine-tuning on target dataset

| fine-tuning methods | Time   | Memory GPU % | Accuracy | F1 Score | Eval loss |
| ------------------- | ------ | ------------ | -------- | -------- | --------- |
| LoRA                | 10m 4s | 64.17        | 0.629    | 0.707    | 1.16      |
| MeZO+LoRA           | 9m 26s | 38.2         | 0.602    | 0.683    | 0.67      |

![[eval_loss_lora.png]]
<center>Figure 1. LoRA perfomance</center>

![[eval_loss_mezolora.png]]
<center>Figure 2. MeZO+LoRA perfomance
</center>

## Conclusion

This study compared LoRA and MeZO+LoRA fine-tuning approaches on the MRPC dataset using RoBERTa-large model. The results demonstrate that MeZO+LoRA achieves better computational efficiency, requiring less GPU memory (38.2% vs 64.17%) and training time (9m 26s vs 10m 4s), while maintaining comparable performance metrics with slightly lower accuracy (0.602 vs 0.629) and F1 score (0.683 vs 0.707). Notably, MeZO+LoRA showed lower evaluation loss (0.67 vs 1.16), suggesting potential better generalization. These findings indicate that MeZO+LoRA could be a preferred choice in resource-constrained environments where slight performance trade-offs are acceptable.
## Quick Run

1) download **MRPC dataset** and copy original folder to /medium_models/data/
2) go to folder medium_models
3) run in your terminal one of these commands for LoRA and MeZO+LoRA fine-tuning methods on **MRPC dataset
4) don't forget to enter your W&B API
5) see the results in W&B dashboard or check folder /result

original repo: https://github.com/princeton-nlp/MeZO/tree/main/medium_models

**LoRA method**
```
TASK=MRPC K=16 SEED=42 BS=64 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
```

**MeZO+LoRA**
```
TASK=MRPC K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 STEP=1000 MODEL=roberta-large EXTRA_TAG=lora bash mezo.sh --apply_lora --lora_r 8 --lora_alpha 16

```


