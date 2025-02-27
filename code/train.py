from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import sys
import json
from datetime import datetime
import argparse
from omegaconf import OmegaConf


def make_dataset(train_data_path_list, validation_data_path_list):
    '''
    데이터셋을 생성하는 함수

    Args:
        train_data_path_list (list): 학습 데이터 파일 경로 리스트
        validation_data_path_list (list): 검증 데이터 파일 경로 리스트
    Returns:
        dataset (DatasetDict): 학습 및 검증 데이터셋
    '''
    loaded_data_dict = {
        'train': {'err_sentence': [], 'cor_sentence': [], 'case': []},
        'validation': {'err_sentence': [], 'cor_sentence': [], 'case': []}
    }

    # 학습 데이터 로드
    for i, train_data_path in enumerate(train_data_path_list):
        with open(train_data_path, 'r') as f:
            _temp_json = json.load(f)
        loaded_data_dict['train']['err_sentence'].extend(
            [x['annotation']['err_sentence'] for x in _temp_json['data']])
        loaded_data_dict['train']['cor_sentence'].extend(
            [x['annotation']['cor_sentence'] for x in _temp_json['data']])
        loaded_data_dict['train']['case'].extend(
            [x['annotation']['case'] for x in _temp_json['data']])
        print(f'train data {i} :', len(_temp_json['data']))

    # 검증 데이터 로드
    for i, validation_data_path in enumerate(validation_data_path_list):
        with open(validation_data_path, 'r') as f:
            _temp_json = json.load(f)
        loaded_data_dict['validation']['err_sentence'].extend(
            [x['annotation']['err_sentence'] for x in _temp_json['data']])
        loaded_data_dict['validation']['cor_sentence'].extend(
            [x['annotation']['cor_sentence'] for x in _temp_json['data']])
        loaded_data_dict['validation']['case'].extend(
            [x['annotation']['case'] for x in _temp_json['data']])
        print(f'validation data {i} :', len(_temp_json['data']))

    dataset_dict = {key: datasets.Dataset.from_dict(value, split=key)
                    for key, value in loaded_data_dict.items()}
    return datasets.DatasetDict(dataset_dict)


def preprocess_function(df, tokenizer, src_col, tgt_col, max_length):
    '''
    데이터 전처리 함수입니다.

    Args:
        df (Dataset): 데이터셋의 데이터
        tokenizer (AutoTokenizer): 모델 토크나이저
        src_col (str): 소스 열 이름
        tgt_col (str): 타겟 열 이름
        max_length (int): 최대 길이
    Returns:
        model_inputs (Dataset): 전처리된 입력 및 라벨 데이터
    '''
    case_map = {
        '1': '[KOR_TO_ENG]',  # 한국어를 영타로 친 경우
        '2': '[ENG_TO_KOR]',  # 영어를 한타로 친 경우
        '3': '[TYPO]'  # 한 글자 틀린 경우
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
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
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
        group_by_length=False,
        report_to=None,
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized['train'],
        eval_dataset=dataset_tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 학습 시작
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Final model saved to {config.output_dir}")


if __name__ == '__main__':
    '''
    Usage:
        python train.py --config-file config/base-config.yaml
    '''
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
