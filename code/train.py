from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import os
import sys
import json
from datetime import datetime
import argparse
from omegaconf import OmegaConf
from datasets import Dataset, DatasetDict, Features, Value
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)

        if labels is not None:
            label_lengths = (labels != -100).sum(dim=1)
            generated_lengths = (logits.argmax(dim=-1) != tokenizer.pad_token_id).sum(dim=1)
            length_penalty = torch.clamp(generated_lengths - label_lengths, min=0).float()
            loss += 0.05 * length_penalty.mean()  # 페널티 가중치 감소

        return (loss, outputs) if return_outputs else loss


def add_typo(sentence):
    if len(sentence) < 2:
        return sentence
    idx = random.randint(0, len(sentence) - 1)
    new_char = chr(random.randint(97, 122))
    return sentence[:idx] + new_char + sentence[idx + 1:]


def make_dataset(train_data_path_list, validation_data_path_list):
    loaded_data_dict = {
        'train': {'err_sentence': [], 'cor_sentence': [], 'case': [], 'score': []},
        'validation': {'err_sentence': [], 'cor_sentence': [], 'case': []}
    }

    for i, train_data_path in enumerate(train_data_path_list):
        with open(train_data_path, 'r') as f:
            _temp_json = json.load(f)
        for x in _temp_json['data']:
            err = x['annotation']['err_sentence']
            cor = x['annotation']['cor_sentence']
            case = str(x['annotation']['case'])
            loaded_data_dict['train']['err_sentence'].append(err)
            loaded_data_dict['train']['cor_sentence'].append(cor)
            loaded_data_dict['train']['case'].append(case)
            loaded_data_dict['train']['score'].append(1.0)

            for _ in range(5):
                if case in ['3']:
                    augmented_err = add_typo(cor)
                    loaded_data_dict['train']['err_sentence'].append(augmented_err)
                    loaded_data_dict['train']['cor_sentence'].append(cor)
                    loaded_data_dict['train']['case'].append(case)
                    loaded_data_dict['train']['score'].append(0.5)
        print(f'train data {i} :', len(_temp_json['data']))

    for i, validation_data_path in enumerate(validation_data_path_list):
        with open(validation_data_path, 'r') as f:
            _temp_json = json.load(f)
        loaded_data_dict['validation']['err_sentence'].extend(
            [x['annotation']['err_sentence'] for x in _temp_json['data']]
        )
        loaded_data_dict['validation']['cor_sentence'].extend(
            [x['annotation']['cor_sentence'] for x in _temp_json['data']]
        )
        loaded_data_dict['validation']['case'].extend(
            [str(x['annotation']['case']) for x in _temp_json['data']]
        )
        print(f'validation data {i} :', len(_temp_json['data']))

    features = Features({
        'err_sentence': Value('string'),
        'cor_sentence': Value('string'),
        'case': Value('string'),
        'score': Value('float32')
    })

    dataset_dict = {
        'train': Dataset.from_dict(loaded_data_dict['train'], split='train', features=features),
        'validation': Dataset.from_dict(loaded_data_dict['validation'], split='validation', features=Features({
            'err_sentence': Value('string'),
            'cor_sentence': Value('string'),
            'case': Value('string')
        }))
    }
    return DatasetDict(dataset_dict)


def preprocess_function(df, tokenizer, src_col, tgt_col, max_length):
    case_map = {
        '1': '[KOR_TO_ENG]',
        '2': '[ENG_TO_KOR]',
        '3': '[TYPO]'
    }
    inputs = [f"{case_map.get(case, '[UNKNOWN]')} {err}" for case, err in zip(df['case'], df[src_col])]
    targets = df[tgt_col]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels['input_ids']
    return model_inputs


def train(config):
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Start ======')
    global tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Finished ======')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print(f'[{_now_time}] ====== Data Load Start ======')
    dataset = make_dataset(config.train_data_path_list, config.validation_data_path_list)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Load Finished ======')

    print(f'[{_now_time}] ====== Data Preprocessing Start ======')
    dataset_tokenized = dataset.map(
        lambda d: preprocess_function(d, tokenizer, config.src_col, config.tgt_col, config.max_length),
        batched=True,
        batch_size=config.per_device_train_batch_size)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Preprocessing Finished ======')

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=40,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
        fp16=False,
        weight_decay=config.weight_decay,
        do_eval=config.do_eval,
        evaluation_strategy="steps",
        log_level="info",
        logging_dir=config.logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        group_by_length=True,
        report_to=None,
        ddp_find_unused_parameters=False,
        label_smoothing_factor=0.1,
    )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized['train'],
        eval_dataset=dataset_tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Final model saved to {config.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    args = parser.parse_args(sys.argv[1:])

    config_file = args.config_file
    config = OmegaConf.load(config_file)

    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Start ==========')

    print(f'DEVICE : cpu')
    print(f'MODEL NAME : {config.pretrained_model_name}')
    print(f'TRAIN FILE PATH :')
    for _path in config.train_data_path_list:
        print(f' - {_path}')
    print(f'VALIDATION FILE PATH :')
    for _path in config.validation_data_path_list:
        print(f' - {_path}')
    print(f'SAVE PATH : {config.output_dir}')
    train(config)

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Finished ==========')
