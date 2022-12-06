# CS224n SQuAD2.0 Project
My solutions to Stanford CS224n NLP default project (Question answering on SQuAD2.0)

## Setup
1. Install requirements
```shell
pip install -r requirements.txt
```
2. Download and transform CS224n SQuAD2.0 (see [data/README.md](https://github.com/zhliuworks/cs224n-squad/tree/master/data))

3. (optional) Download pretrained weights and put them in `weight/` just for evaluation. Here is a list of model pretrained weights ranked by #parameters.

| Model           | Parameters (M) | FLOPs (G) | Download                                  |
| --------------- | -------------- | --------- | ----------------------------------------- |
| ALBERT-base     | 7.19           | 32.68     | [link](https://jbox.sjtu.edu.cn/l/b1x4Ww) |
| DeBERTaV3-base  | 85.06          | 39.89     | [link](https://jbox.sjtu.edu.cn/l/4HrIIM) |
| ALBERT-xxlarge  | 201.92         | 928.07    | [link](https://jbox.sjtu.edu.cn/l/M1A9Ro) |
| DeBERTaV3-large | 302.32         | 141.81    | [link](https://jbox.sjtu.edu.cn/l/l1xKkd) |

## Train
Here is an example to finetune ALBERT-base model with FP16 training.
```shell
python train.py \
--pretrain_model albert-base-v2 \
--batch_size 16 \
--gpu_ids 0,1 \
--num_epochs 10 \
--lr 1e-4 \
--weight_decay 1e-4 \
--use_fp16
```

## Test
Here is an example to evaluate a trained ALBERT-base model.
```shell
python test.py \
--pretrain_model albert-base-v2 \
--trained_weight_path weight/albert-base-v2.pth \
--batch_size 16 \
--gpu_ids 2
```

## Results
| Model                                               |                                                              | F1        | EM        |
| --------------------------------------------------- | ------------------------------------------------------------ | --------- | --------- |
| [BiDAF](https://arxiv.org/abs/1611.01603)           | a weak Non-PCE baseline                                      | 61.59     | 58.29     |
| [ALBERT-base](https://arxiv.org/abs/1909.11942)     | finetuned on 🤗HF [albert-base-v2](https://huggingface.co/albert-base-v2) | 81.35     | 78.64     |
| [DeBERTaV3-base](https://arxiv.org/abs/2111.09543)  | finetuned on 🤗HF [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | 86.89     | 83.35     |
| [ALBERT-xxlarge](https://arxiv.org/abs/1909.11942)  | finetuned on 🤗HF [albert-xxlarge-v1](https://huggingface.co/albert-xxlarge-v1) | 88.67     | 85.75     |
| [DeBERTaV3-large](https://arxiv.org/abs/2111.09543) | finetuned on 🤗HF [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) | **90.42** | **87.36** |

## References
[1] https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb

[2] https://github.com/chrischute/squad
