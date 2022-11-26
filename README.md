# CS224n SQuAD2.0 Project
My solutions to Stanford CS224n NLP default project (Question answering on SQuAD2.0)

TODO: update the requirements.txt via `pipreqs .`

## Setup
1. Install requirements
```shell
pip install -r requirements.txt
```
2. Download and transform CS224n-specific SQuAD2.0 (see [data/README.md](https://github.com/zhliuworks/cs224n-squad/tree/master/data))

## Train
...

## Test
* Test [elgeish/cs224n-squad2.0-albert-base-v2](https://huggingface.co/elgeish/cs224n-squad2.0-albert-base-v2)
```shell
python test.py --ckpt_path elgeish/cs224n-squad2.0-albert-base-v2
```

## Results
| Model                                           |                                                              | F1    | EM    |
| ----------------------------------------------- | ------------------------------------------------------------ | ----- | ----- |
| [BiDAF](https://arxiv.org/abs/1611.01603)       | a weak Non-PCE baseline                                      | 61.59 | 58.29 |
| [ALBERT-base](https://arxiv.org/abs/1909.11942) | [elgeish's](https://huggingface.co/elgeish/cs224n-squad2.0-albert-base-v2), based on HF albert-base-v2 | 81.20 | 78.48 |

## References
[1] https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb

[2] https://github.com/chrischute/squad
