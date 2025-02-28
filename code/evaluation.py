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
    return 1.0 if prd_sentence == cor_sentence else 0.0

def calc_edit_distance(cor_sentence, prd_sentence):
    return Levenshtein.distance(cor_sentence, prd_sentence)

def calc_char_accuracy(cor_sentence, prd_sentence):
    if not cor_sentence:
        return 0.0
    matches = sum(1 for c1, c2 in zip(cor_sentence, prd_sentence) if c1 == c2)
    return matches / len(cor_sentence)

def remove_repetition(sentence):
    words = sentence.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)

def my_train(gpus='cpu', model_path=None, test_file=None, eval_length=None, save_path=None, pb=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_datasets(test_file)

    device = torch.device(gpus)
    model.to(device)

    id_list, err_sentence_list, cor_sentence_list = [], [], []
    prd_sentence_list_1, prd_sentence_list_2, prd_sentence_list_3 = [], [], []
    accuracy_list_1, accuracy_list_2, accuracy_list_3 = [], [], []
    edit_distance_list_1, edit_distance_list_2, edit_distance_list_3 = [], [], []
    char_accuracy_list_1, char_accuracy_list_2, char_accuracy_list_3 = [], [], []

    data_len = len(dataset['test'])
    if eval_length:
        data_len = min(eval_length, data_len)

    print('=' * 100)
    for n in tqdm(range(data_len), disable=pb):
        data_id = dataset['test'][n]['id']
        err_sentence = dataset['test'][n]['err_sentence']
        cor_sentence = dataset['test'][n]['cor_sentence']

        cor_words = cor_sentence.split()
        cor_word_count = len(cor_words)
        cor_length = len(tokenizer.tokenize(cor_sentence))

        tokenized = tokenizer(err_sentence, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)

        max_length = cor_length + 3
        min_length = max(cor_length - 2, 1)

        res = model.generate(
            inputs=input_ids,
            num_beams=10,
            num_return_sequences=3,  # Generate 3 candidates
            temperature=1.0,
            repetition_penalty=2.0,
            length_penalty=0.8,
            no_repeat_ngram_size=2,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        ).cpu().tolist()

        prd_sentences = []
        for i in range(min(3, len(res))):  # Handle cases where fewer than 3 are returned
            prd_sentence = tokenizer.decode(res[i], skip_special_tokens=True).strip()
            prd_sentence = remove_repetition(prd_sentence)
            prd_words = prd_sentence.split()
            prd_sentence = ' '.join(prd_words[:cor_word_count])
            prd_sentences.append(prd_sentence)

        while len(prd_sentences) < 3:
            prd_sentences.append("")

        accuracy_1 = calc_accuracy(cor_sentence, prd_sentences[0])
        accuracy_2 = calc_accuracy(cor_sentence, prd_sentences[1])
        accuracy_3 = calc_accuracy(cor_sentence, prd_sentences[2])

        edit_distance_1 = calc_edit_distance(cor_sentence, prd_sentences[0])
        edit_distance_2 = calc_edit_distance(cor_sentence, prd_sentences[1])
        edit_distance_3 = calc_edit_distance(cor_sentence, prd_sentences[2])

        char_accuracy_1 = calc_char_accuracy(cor_sentence, prd_sentences[0])
        char_accuracy_2 = calc_char_accuracy(cor_sentence, prd_sentences[1])
        char_accuracy_3 = calc_char_accuracy(cor_sentence, prd_sentences[2])

        id_list.append(data_id)
        err_sentence_list.append(err_sentence)
        cor_sentence_list.append(cor_sentence)
        prd_sentence_list_1.append(prd_sentences[0])
        prd_sentence_list_2.append(prd_sentences[1])
        prd_sentence_list_3.append(prd_sentences[2])
        accuracy_list_1.append(accuracy_1)
        accuracy_list_2.append(accuracy_2)
        accuracy_list_3.append(accuracy_3)
        edit_distance_list_1.append(edit_distance_1)
        edit_distance_list_2.append(edit_distance_2)
        edit_distance_list_3.append(edit_distance_3)
        char_accuracy_list_1.append(char_accuracy_1)
        char_accuracy_list_2.append(char_accuracy_2)
        char_accuracy_list_3.append(char_accuracy_3)

        _cnt = n + 1
        _per_calc = round(_cnt / data_len, 4)
        _now_time = datetime.now().__str__()
        print(f'[{_now_time}] - [{_per_calc:6.1%} {_cnt:06,}/{data_len:06,}] - Evaluation Result (Data id : {data_id})')
        print(f'{" " * 30} >       TEST : {err_sentence}')
        print(f'{" " * 30} >    PREDICT 1 : {prd_sentences[0]}')
        print(f'{" " * 30} >    PREDICT 2 : {prd_sentences[1]}')
        print(f'{" " * 30} >    PREDICT 3 : {prd_sentences[2]}')
        print(f'{" " * 30} >      LABEL : {cor_sentence}')
        print(f'{" " * 30} > ACCURACY 1 : {accuracy_1:6.3f}')
        print(f'{" " * 30} > ACCURACY 2 : {accuracy_2:6.3f}')
        print(f'{" " * 30} > ACCURACY 3 : {accuracy_3:6.3f}')
        print(f'{" " * 30} > EDIT DISTANCE 1 : {edit_distance_1}')
        print(f'{" " * 30} > EDIT DISTANCE 2 : {edit_distance_2}')
        print(f'{" " * 30} > EDIT DISTANCE 3 : {edit_distance_3}')
        print(f'{" " * 30} > CHAR ACCURACY 1 : {char_accuracy_1:6.3f}')
        print(f'{" " * 30} > CHAR ACCURACY 2 : {char_accuracy_2:6.3f}')
        print(f'{" " * 30} > CHAR ACCURACY 3 : {char_accuracy_3:6.3f}')
        print('=' * 100)

        torch.cuda.empty_cache()

    save_file_name = os.path.split(test_file)[-1].replace('.json', '') + '.csv'
    save_file_path = os.path.join(save_path, save_file_name)
    df = pd.DataFrame({
        'id': id_list,
        'err_sentence': err_sentence_list,
        'prd_sentence_1': prd_sentence_list_1,
        'prd_sentence_2': prd_sentence_list_2,
        'prd_sentence_3': prd_sentence_list_3,
        'cor_sentence': cor_sentence_list,
        'accuracy_1': accuracy_list_1,
        'accuracy_2': accuracy_list_2,
        'accuracy_3': accuracy_list_3,
        'edit_distance_1': edit_distance_list_1,
        'edit_distance_2': edit_distance_list_2,
        'edit_distance_3': edit_distance_list_3,
        'char_accuracy_1': char_accuracy_list_1,
        'char_accuracy_2': char_accuracy_list_2,
        'char_accuracy_3': char_accuracy_list_3
    })
    df.to_csv(save_file_path, index=True)
    print(f'[{datetime.now()}] - Save Result File(.csv) - {save_file_path}')

    print('=' * 100)
    mean_accuracy_1 = sum(accuracy_list_1) / len(accuracy_list_1)
    mean_accuracy_2 = sum(accuracy_list_2) / len(accuracy_list_2)
    mean_accuracy_3 = sum(accuracy_list_3) / len(accuracy_list_3)
    mean_edit_distance_1 = sum(edit_distance_list_1) / len(edit_distance_list_1)
    mean_edit_distance_2 = sum(edit_distance_list_2) / len(edit_distance_list_2)
    mean_edit_distance_3 = sum(edit_distance_list_3) / len(edit_distance_list_3)
    mean_char_accuracy_1 = sum(char_accuracy_list_1) / len(char_accuracy_list_1)
    mean_char_accuracy_2 = sum(char_accuracy_list_2) / len(char_accuracy_list_2)
    mean_char_accuracy_3 = sum(char_accuracy_list_3) / len(char_accuracy_list_3)
    print(f'       Average Accuracy 1 : {mean_accuracy_1:6.3f}')
    print(f'       Average Accuracy 2 : {mean_accuracy_2:6.3f}')
    print(f'       Average Accuracy 3 : {mean_accuracy_3:6.3f}')
    print(f'       Average Edit Distance 1 : {mean_edit_distance_1:6.3f}')
    print(f'       Average Edit Distance 2 : {mean_edit_distance_2:6.3f}')
    print(f'       Average Edit Distance 3 : {mean_edit_distance_3:6.3f}')
    print(f'       Average Char Accuracy 1 : {mean_char_accuracy_1:6.3f}')
    print(f'       Average Char Accuracy 2 : {mean_char_accuracy_2:6.3f}')
    print(f'       Average Char Accuracy 3 : {mean_char_accuracy_3:6.3f}')
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
    print(f'DEVICE : {gpu_no}, MODEL PATH : {args.model_path}, FILE PATH : {args.test_file}, DATA LENGTH : {args.eval_length}, SAVE PATH : {save_path}')
    my_train(gpu_no, model_path=args.model_path, test_file=args.test_file, eval_length=args.eval_length,
             save_path=save_path, pb=pb)
    print(f'[{datetime.now()}] ========== Evaluation Finished ==========')