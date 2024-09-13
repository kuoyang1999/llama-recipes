import os
import tqdm
import json
import copy
import math

import torch
import logging
import argparse
from datetime import datetime

import numpy as np
from rouge import Rouge

import dataclasses
from xopen import xopen

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets.dataset import load_data

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':

    # set up args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="~/Projects/models/LLMs/Llama-2-7b-chat-hf")
    parser.add_argument("--sample_num", type=int)
    parser.add_argument("--dataset", type=str, choices=["xsum", "cnn_dailymail", "iclr", "nips"], default="xsum")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # set seed
    set_seed(args)

    # load model
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,)
    
    # load dataset
    dataset_name = args.dataset
    responses = load_data(dataset_name)

    # control sample data number
    if args.sample_num and args.sample_num < len(responses):
        responses = responses[:args.sample_num]
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(responses)))

    # load rouge score
    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    with torch.no_grad():
        for response in tqdm.tqdm(responses):
            result, prompt, label, temperature, stop, top_p, max_tokens, n = response

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=max_tokens + len(input_ids[0]),
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=n,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            scores = rouge.get_scores(generate_text, label)[0]
            rouge1_score_list.append(scores['rouge-1']['f'])
            rouge2_score_list.append(scores['rouge-2']['f'])
            rougel_score_list.append(scores['rouge-l']['f'])

            result['result'] = generate_text
            
            result['analysis']= {
                'prompt': prompt,
                'evaluation': {
                    'rouge1': scores['rouge-1']['f'],
                    'rouge2': scores['rouge-2']['f'],
                    'rougel': scores['rouge-l']['f']
                },
                "logprobs": {
                    "tokens": tokens, 
                    "token_logprobs": logprobs, 
                    "top_logprobs": top_logprobs, 
                    "text_offset": []
                }, 
            }
            
            results.append(result)

    print('Average Rouge1: {:.6f}, Rouge-2: {:.6f}, Rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
    # add experiment setting and average rouge score to the start of the json file
    experiment_info = {
        'experiment_setting': {
            'model_name': model_name,
            'sample_num': args.sample_num,
            'dataset': dataset_name,
            'seed': args.seed
        },
        'average_rouge_score': {
            'rouge1': np.mean(rouge1_score_list),
            'rouge2': np.mean(rouge2_score_list),
            'rougel': np.mean(rougel_score_list)
        }
    }
    results.insert(0, experiment_info)
    # Define the directory path
    output_dir = f"recipes/experimental/multi_document_summarization/data/summarization_output/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the full file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/{timestamp}.json"   
    # output_path = f"{output_dir}/{dataset_name}_{timestamp}.json"   

    # Write the results to the file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


