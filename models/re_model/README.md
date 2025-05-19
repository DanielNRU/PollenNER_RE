---
library_name: transformers
base_model: DeepPavlov/rubert-base-cased
tags:
- generated_from_trainer
metrics:
- f1
model-index:
- name: pollen-re-model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pollen-re-model

This model is a fine-tuned version of [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3497
- F1: 0.9149

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1     |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| No log        | 1.0   | 340  | 0.5049          | 0.5574 |
| 0.4748        | 2.0   | 680  | 0.3497          | 0.9149 |
| 0.3114        | 3.0   | 1020 | 0.5014          | 0.9082 |
| 0.3114        | 4.0   | 1360 | 0.5509          | 0.8393 |


### Framework versions

- Transformers 4.51.3
- Pytorch 2.7.0+cu128
- Datasets 3.5.0
- Tokenizers 0.21.1
