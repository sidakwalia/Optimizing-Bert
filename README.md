# Optimized BERT - Pretraining and Fine-tuning

This repository is dedicated to an optimized BERT implementation, where we used Wikipedia dataset from Hugging Face and divided it into 2560 Hadoop files. The repository provides tools for both pre-training BERT from scratch and fine-tuning the pre-trained models.

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
We have fine-tuned our model on the MRPC dataset. You can find the fine-tuning results in the `tmp/finetune` folder. Run the bash `finetune.sh` for fine-tuning the dataset. The script supports various tasks like MRPC, CoLA, MNLI, QNLI, and more
