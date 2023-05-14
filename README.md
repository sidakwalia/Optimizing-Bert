# Optimized BERT - Pretraining and Fine-tuning

This repository is a part of the HPML Final Project by Inder Preet Singh Walia and Jash Rathod. It is an optimized BERT implementation, where we used the Wikipedia dataset from Hugging Face and divided it into 2560 Hadoop files. The repository provides tools for both pre-training BERT from scratch and fine-tuning the pre-trained models.

## Dataset Preparation
To download and prepare the dataset, run `prepare_dataset.py`. This will download the Wikipedia dataset and shard it into Hadoop files.

## Pretraining
For pre-training, run the bash `optimizer.sh` file. It uses different optimizers like Adam, Lamb, AdamW, and Adafactor. You can adjust the number of epochs using the parameter in the bash file.

The script `run_pretraining.py` is used for the pretraining process, which accepts a number of parameters:

- `model_type` and `tokenizer_name`: These specify the model type and tokenizer (here, 'bert-large-uncased').

- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`: These are hyperparameters for the architecture of the BERT model.

- `lr`, `train_batch_size`, `train_micro_batch_size_per_gpu`, `lr_schedule`, `curve`, `warmup_proportion`, `gradient_clipping`: These are various parameters to control the learning process.

- `optimizer_type`, `weight_decay`, `adam_beta1`, `adam_beta2`, `adam_eps`: These parameters control the optimizer behavior.

- `total_training_time`, `early_exit_time_marker`, `dataset_path`, `output_dir`, `print_steps`, `num_epochs_between_checkpoints`: These parameters help in managing the training process, data locations and checkpointing.

- `job_name`, `project_name`, `validation_epochs`, `validation_epochs_begin`, `validation_epochs_end`, `validation_begin_proportion`, `validation_end_proportion`, `validation_micro_batch`: These parameters are for managing the validation process and logging.

- `deepspeed`, `data_loader_type`, `do_validation`, `use_early_stopping`, `early_stop_time`, `early_stop_eval_loss`, `seed`: These parameters are for using deepspeed, data loading, validation, early stopping, and setting the random seed.

## Fine-tuning
We have fine-tuned our model on the MRPC dataset. You can find the fine-tuning results in the `tmp/finetune` folder. Run the bash `finetune.sh` for fine-tuning the dataset. The script supports various tasks like MRPC, CoLA, MNLI, QNLI, and more.

The `run_glue.py` script is used for the fine-tuning process, which accepts a number of parameters:

- `model_name_or_path`: The path to the model or model type to use for fine-tuning.

- `task_name`: The GLUE task to be used for fine-tuning.

- `max_seq_length`: The maximum total input sequence length after tokenization.

- `output_dir`: The directory where the model predictions and checkpoints will be written.

- `do_train`, `do_eval`: Flags to determine whether to perform training and evaluation.

- `evaluation_strategy`: The evaluation strategy to use.

- `per_device_train_batch_size`, `per_device_eval_batch_size`, `gradient_accumulation_steps`: These parameters control the batch size for training and evaluation.

- `learning_rate`, `weight_decay`, `max_grad_norm`: These parameters control the learning behavior.

- `eval_steps`: The number of steps between evaluations.

- `num_train_epochs`: The total number of training epochs to perform.

- `lr_scheduler_type`: The learning rate scheduler type to use.

## Results
In our experiments, the following conclusions were made:

- **1000 epochs**: AdamW optimizer and fp32 quantization achieved the lowest fine-tune loss and the highest fine-tune accuracy and F1 score.
- **4000 epochs**: AdamW optimizer and fp16 quantization achieved the highest fine-tune accuracy and F1 score, even though it did not have the lowest fine-tune loss (training for more steps needed).
- Longer training improves results.
- The LAMB optimizer did not yield the highest performance in these experiments, contrary to what might be expected given its design for larger models and larger batch sizes.

## References
We have referred to the following repository for our project: [IntelLabs' academic-budget-bert](https://github.com/IntelLabs/academic-budget-bert)

## Contributing
Please feel free to open an issue or pull request if you find a bug or have suggestions for improvements.

## License
This project is licensed under the terms of the MIT license.
