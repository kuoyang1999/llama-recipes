import os
import json

def load_data(dataset_name):
    # Request: raw json data
    # Response: processed data (result, prompt, label, temperature, stop, top_p, max_tokens)
    
    # input path
    input_path = f"recipes/experimental/multi_document_summarization/datasets/{dataset_name}/{dataset_name}.json"
    
    with open(input_path, 'r') as f:
        requests = json.load(f)
    
    if dataset_name == "xsum":
        return load_xsum_data(requests)
    elif dataset_name == "cnn_dailymail":
        return load_cnn_dailymail_data(requests)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_xsum_data(requests):
    responses = []
    for request in requests:
        prompt = request['article']
        label = request['summary_gt'] # ground truth summary
        temperature = request['temperature']
        stop = request['stop']
        top_p = request['top_p']
        max_tokens = request['max_tokens']
        n = request['n']
        result = {'request': request['article'], 'lable': label, 'result': {}} # request is the document that needs to be summarized, result is the summary
        responses.append((result, prompt, label, temperature, stop, top_p, max_tokens, n))
    return responses

def load_cnn_dailymail_data(requests):
    responses = []
    for request in requests:
        prompt = request['story']
        label = request['highlights']
        temperature = request['temperature']
        stop = request['stop']
        top_p = request['top_p']
        max_tokens = request['max_tokens']
        n = request['n']
        result = {'request': request['article'], 'lable': label, 'result': {}} # request is the document that needs to be summarized, result is the summary
        responses.append((result, prompt, label, temperature, stop, top_p, max_tokens, n))
    return responses