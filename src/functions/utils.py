import logging
import random
import torch
import numpy as np
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.functions.processor import (
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)

# from transformers.data.processors.squad import (
#     SquadV1Processor,
#     SquadV2Processor,
#     squad_convert_examples_to_features
# )

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# tensor를 list 형으로 변환하기위한 함수
def to_list(tensor):
    return tensor.detach().cpu().tolist()

# dataset을 load 하는 함수
from torch.utils.data import TensorDataset
def load_examples(args, tokenizer, evaluate=False, output_examples=False, do_cache=False, data=None):
    '''

    :param args: 하이퍼 파라미터
    :param tokenizer: tokenization에 사용되는 tokenizer
    :param evaluate: 평가나 open test시, True
    :param output_examples: 평가나 open test 시, True / True 일 경우, examples와 features를 같이 return
    :return:
    examples : max_length 상관 없이, 원문으로 각 데이터를 저장한 리스트
    features : max_length에 따라 분할 및 tokenize된 원문 리스트
    dataset : max_length에 따라 분할 및 학습에 직접적으로 사용되는 tensor 형태로 변환된 입력 ids
    '''
    if do_cache:
        input_dir = args.train_file_path if not evaluate else args.dev_file_path
        cached_features_file = os.path.join(
            args.train_cache_path if not evaluate else args.dev_cache_path,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                data.split(".")[1],
                str(args.max_seq_length),
            ))
        print("Creating features from dataset file at {}".format(input_dir))

        # processor 선언
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

        # open test 시
        if evaluate:
            examples = processor.get_dev_examples(os.path.join(args.dev_file_path),
                                                  filename=data)
        # 학습 시
        else:
            examples = processor.get_train_examples(os.path.join(args.train_file_path),
                                                    filename=data)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
    else:
        if evaluate:
            input_dir = args.dev_cache_path
        else:
            input_dir = args.train_cache_path
        cached_features_file = os.path.join(
            input_dir,
            data,
        )
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )

        if output_examples:
            return dataset, examples, features

        all_input_ids = []
        all_attention_masks = []
        all_question_masks = []
        all_sentence_masks = []
        all_token_type_ids = []
        all_question_type_label = []
        all_sent_start_positions = []
        all_sent_end_positions = []
        all_tok_start_positions = []
        all_tok_end_positions = []

        for idx in range(len(features)):
            feature = features[idx]
            example = examples[feature.example_index]
            if (feature.tok_end_position == 0 and feature.sent_end_position == 0):
                continue

            all_input_ids.append(dataset.tensors[0][idx])

            all_attention_masks.append(dataset.tensors[1][idx])
            all_question_masks.append(dataset.tensors[2][idx])
            all_sentence_masks.append(dataset.tensors[3][idx])
            all_token_type_ids.append(dataset.tensors[4][idx])
            all_question_type_label.append(dataset.tensors[5][idx])
            all_sent_start_positions.append(dataset.tensors[6][idx])
            all_sent_end_positions.append(dataset.tensors[7][idx])
            all_tok_start_positions.append(dataset.tensors[8][idx])
            all_tok_end_positions.append(dataset.tensors[9][idx])
        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_attention_masks = torch.stack(all_attention_masks, dim=0)
        all_question_masks = torch.stack(all_question_masks, dim=0)
        all_sentence_masks = torch.stack(all_sentence_masks, dim=0)
        all_token_type_ids = torch.stack(all_token_type_ids, dim=0)
        all_question_type_label = torch.stack(all_question_type_label, dim=0)
        all_sent_start_positions = torch.stack(all_sent_start_positions, dim=0)
        all_sent_end_positions = torch.stack(all_sent_end_positions, dim=0)
        all_tok_start_positions = torch.stack(all_tok_start_positions, dim=0)
        all_tok_end_positions = torch.stack(all_tok_end_positions, dim=0)
        new_dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_question_masks,
            all_sentence_masks,
            all_token_type_ids,
            all_question_type_label,
            all_sent_start_positions,
            all_sent_end_positions,
            all_tok_start_positions,
            all_tok_end_positions,
        )
        return new_dataset
def precision_measure(labels, preds, num_label):
    c, m = [0]*num_label, [0]*num_label
    for idx, label in enumerate(preds):
        if label == labels[idx]:
            c[label]+=1
        m[label]+=1
    return [e/m[idx] if m[idx] !=0 else 0.0 for idx, e in enumerate(c)]

def recall_measure(labels, preds, num_label):
    c, m = [0] * num_label, [0] * num_label
    for idx, label in enumerate(labels):
        if label == preds[idx]:
            c[label] += 1
        m[label] += 1
    return [e / m[idx] if m[idx] != 0 else 0.0 for idx, e in enumerate(c)]

def f1_measure(labels, preds, num_label):
    recalls = recall_measure(labels, preds, num_label)
    precisions = precision_measure(labels, preds, num_label)

    f1_scores = [(precisions[idx]*recalls[idx]*2)/(precisions[idx]+recalls[idx]) if precisions[idx]+recalls[idx] != 0 else 0 for idx in range(len(recalls))]

    print(f1_scores)
    print(np.mean(f1_scores))