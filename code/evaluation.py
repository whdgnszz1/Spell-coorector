from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import datasets
import pandas as pd
import os
import sys
import json
from tqdm import tqdm
from datetime import datetime
import argparse
import Levenshtein


def load_datasets(test_file):
    with open(test_file, 'r') as f:
        json_dataset = json.load(f)

    list_dataset = {
        'id': [data['metadata_info']['id'] for data in json_dataset['data']],
        'err_sentence': [data['annotation']['err_sentence'] for data in json_dataset['data']],
        'cor_sentence': [data['annotation']['cor_sentence'] for data in json_dataset['data']]
    }

    dataset_dict = {
        'test': datasets.Dataset.from_dict(list_dataset, split='test')
    }
    return datasets.DatasetDict(dataset_dict)


def calc_accuracy(cor_sentence, prd_sentence):
    """예측 문장과 정답 문장이 완전히 일치하면 1.0, 아니면 0.0"""
    return 1.0 if prd_sentence == cor_sentence else 0.0


def calc_edit_distance(cor_sentence, prd_sentence):
    """정답 문장과 예측 문장 간의 편집 거리"""
    return Levenshtein.distance(cor_sentence, prd_sentence)


def calc_char_accuracy(cor_sentence, prd_sentence):
    """문자 단위 정확도: 일치하는 문자 수 / 정답 문장 길이"""
    if not cor_sentence:
        return 0.0
    matches = sum(1 for c1, c2 in zip(cor_sentence, prd_sentence) if c1 == c2)
    return matches / len(cor_sentence)


def my_train(gpus='cpu', model_path=None, test_file=None, eval_length=None, save_path=None, pb=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_datasets(test_file)

    device = torch.device(gpus)
    model.to(device)

    id_list, err_sentence_list, cor_sentence_list, prd_sentence_list = [], [], [], []
    accuracy_list, edit_distance_list, char_accuracy_list = [], [], []

    data_len = len(dataset['test'])
    if eval_length:
        data_len = min(eval_length, data_len)

    print('=' * 100)
    for n in tqdm(range(data_len), disable=pb):
        data_id = dataset['test'][n]['id']
        err_sentence = dataset['test'][n]['err_sentence']
        cor_sentence = dataset['test'][n]['cor_sentence']

        tokenized = tokenizer(err_sentence, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)
        res = model.generate(
            inputs=input_ids,
            num_beams=20,
            num_return_sequences=2,
            temperature=2,
            repetition_penalty=0.2,
            length_penalty=0.2,
            no_repeat_ngram_size=2,
            max_length=input_ids.size()[1] + 5
        ).cpu().tolist()[0]
        prd_sentence = tokenizer.decode(res).replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()

        id_list.append(data_id)
        err_sentence_list.append(err_sentence)
        cor_sentence_list.append(cor_sentence)
        prd_sentence_list.append(prd_sentence)

        # 정확도, 편집 거리, 문자 단위 정확도 계산
        accuracy = calc_accuracy(cor_sentence, prd_sentence)
        edit_distance = calc_edit_distance(cor_sentence, prd_sentence)
        char_accuracy = calc_char_accuracy(cor_sentence, prd_sentence)
        accuracy_list.append(accuracy)
        edit_distance_list.append(edit_distance)
        char_accuracy_list.append(char_accuracy)

        _cnt = n + 1
        _per_calc = round(_cnt / data_len, 4)
        _now_time = datetime.now().__str__()
        print(f'[{_now_time}] - [{_per_calc:6.1%} {_cnt:06,}/{data_len:06,}] - Evaluation Result (Data id : {data_id})')
        print(f'{" " * 30} >       TEST : {err_sentence}')
        print(f'{" " * 30} >    PREDICT : {prd_sentence}')
        print(f'{" " * 30} >      LABEL : {cor_sentence}')
        print(f'{" " * 30} > ACCURACY : {accuracy:6.3f}')
        print(f'{" " * 30} > EDIT DISTANCE : {edit_distance}')
        print(f'{" " * 30} > CHAR ACCURACY : {char_accuracy:6.3f}')
        print('=' * 100)

        torch.cuda.empty_cache()

    save_file_name = os.path.split(test_file)[-1].replace('.json', '') + '.csv'
    save_file_path = os.path.join(save_path, save_file_name)
    df = pd.DataFrame({
        'id': id_list,
        'err_sentence': err_sentence_list,
        'prd_sentence': prd_sentence_list,
        'cor_sentence': cor_sentence_list,
        'accuracy': accuracy_list,
        'edit_distance': edit_distance_list,
        'char_accuracy': char_accuracy_list
    })
    df.to_csv(save_file_path, index=True)
    print(f'[{datetime.now()}] - Save Result File(.csv) - {save_file_path}')

    print('=' * 100)
    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    mean_edit_distance = sum(edit_distance_list) / len(edit_distance_list)
    mean_char_accuracy = sum(char_accuracy_list) / len(char_accuracy_list)
    print(f'       Average Accuracy : {mean_accuracy:6.3f}')
    print(f'       Average Edit Distance : {mean_edit_distance:6.3f}')
    print(f'       Average Char Accuracy : {mean_char_accuracy:6.3f}')
    print('=' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_no", dest="gpu_no", type=int, action="store")
    parser.add_argument("--model_path", dest="model_path", type=str, action="store")
    parser.add_argument("--test_file", dest="test_file", type=str, action="store")
    parser.add_argument("--eval_length", dest="eval_length", type=int, action="store")
    parser.add_argument("-pb", dest="pb", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    gpu_no = 'cpu'
    if args.gpu_no is not None:
        gpu_no = f'cuda:{args.gpu_no}'

    pb = not args.pb

    print(f'[{datetime.now()}] ========== Evaluation Start ==========')
    print(
        f'DEVICE : {gpu_no}, MODEL PATH : {args.model_path}, FILE PATH : {args.test_file}, DATA LENGTH : {args.eval_length}, SAVE PATH : {save_path}')
    my_train(gpu_no, model_path=args.model_path, test_file=args.test_file, eval_length=args.eval_length,
             save_path=save_path, pb=pb)
    print(f'[{datetime.now()}] ========== Evaluation Finished ==========')
