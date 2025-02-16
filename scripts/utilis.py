import os
import re
import json
import uuid
from Levenshtein import distance
import csv
import fire

def load_jsonl(path):
    if path.endswith('json'):
        with open(path, 'r', encoding='utf-8') as fr:
            data=json.load(fr)
        
    else:
        data = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path,mode='w',**kwargs):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False,**kwargs) + '\n')

def find_non_utf8_lines(filename):
    non_utf8_lines = []
    with open(filename, 'rb') as file:
        line_number = 0
        for line in file:
            line_number += 1
            try:
                decoded_line = line.decode('utf-8')
                json.loads(decoded_line)
            except UnicodeDecodeError:
                non_utf8_lines.append(line_number)
            except json.JSONDecodeError:
                print(f"Line {line_number} is not valid JSON.")
    return non_utf8_lines

            
def txt2list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        my_list = [line.strip() for line in lines]
    return my_list

def list2txt(id_list,output_file):
    with open(output_file, 'w') as f:
        for item in id_list:
            f.write(f"{item}\n")

def get_prediction(output):
    pattern = r"score is \(?([abcdefghij])\)?"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    else:
        
        return output

def get_judgement(answer):
    match1 = re.search(r'Judgement":\s*(.*?)$', answer,re.DOTALL)
    if match1: 
        answer = match1.group(1).strip()
        # answer = answer.replace('\\', '\\\\')
    else:
        answer=answer
    return answer

def get_full_judgement_w_None(answer):
    match1 = re.search(r'Judgement":\s*(.*?)(?=,\s*"Score)', answer,re.DOTALL)
    match2 = re.search(r'Score":\s*(.*?)\n}\n', answer,re.DOTALL)
    match3 = re.search(r'Score":\s*(.*?)\n', answer,re.DOTALL)
    if match1 and (match2 or match3): 
        Judgement = match1.group(1).strip()
        if match2:
            score = match2.group(1).strip()
        else:
            score = match3.group(1).strip()
        score = score.replace("\"","")
        returndict={
            "Score":score,
            "Judgement":Judgement,
            "id":str(uuid.uuid4())
        }
        # answer = answer.replace('\\', '\\\\')
    else:
        returndict=None
    return returndict


def get_score_first(answer):

    # match1 = re.search(r'Score":\s*(.*?)(?=,\s*"Judgement)', answer,re.DOTALL)
    # match2 = re.search(r'Score:\s*(.*?)(?=\s*Judgement)', answer,re.DOTALL)
    match1 = re.search(r'Score":\s*(.*?)\n}\n', answer,re.DOTALL)
    match2 = re.search(r'Score":\s*(.*?)\n', answer,re.DOTALL)
    if match1:
        answer = match1.group(1).strip()
        # answer = answer.replace('\\', '\\\\')
        try:
            if answer.startswith('"') and answer.endswith('"'):
                   answer = json.loads(answer)
        except json.JSONDecodeError as e:
                print(f"Failed to parse JSON due to: {e}")
                answer = None
    else:
        if match2:
            answer = match2.group(1).strip()
            # answer = answer.replace('\\', '\\\\')
            try:
                if answer.startswith('"') and answer.endswith('"'):
                   answer = json.loads(answer)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON due to: {e}")
                answer = None
        else:
            answer = get_prediction(answer)
    if answer is not None:
        try:
            answer=float(answer)
            answer= 5 if answer>5 else answer
            answer= 0 if answer<0 else answer
        except:
            answer = None
        
    return answer
def sort_index_with_none(scores):
    clean_list = [(i,x) for i,x in enumerate(scores) if x is not None]
    sorted_list= sorted(clean_list, key=lambda x:x[1],reverse=True)
    sorted_index=[t[0] for t in sorted_list]
    return sorted_index

def find_max_index_with_none(lst):
    clean_list = [x for x in lst if x is not None]
    max_value = max(clean_list)
    max_index = next((i for i, x in enumerate(lst) if x == max_value), None)
    return max_index

def find_min_index_with_none(lst):
    clean_list = [x for x in lst if x is not None]
    min_value = min(clean_list)
    min_index = next((i for i, x in enumerate(lst) if x == min_value), None)
    return min_index

def find_LC_index_with_none(scores,responses,index):
    clean_list = [(i,x) for i,x in enumerate(scores) if x is not None]
    assert 0<=index and index<=1
    _,max_value = max(clean_list, key=lambda x: x[1])
    _,min_value = min(clean_list, key=lambda x: x[1])
    max_threshold = max_value*(1-index)+min_value*index
    min_threshold = min_value*(1-index)+max_value*index
    max_indexes=[i for i, x in clean_list if x <= max_value and x >= max_threshold]
    min_indexes=[i for i, x in clean_list if x >= min_value and x <= min_threshold]
    max_lengths = ((index, len(responses[index][1])) for index in max_indexes)
    max_shortest_index, _ = min(max_lengths, key=lambda x: x[1], default=(None, None))
    min_lengths = ((index, len(responses[index][1])) for index in min_indexes)
    min_longest_index, _ = max(min_lengths, key=lambda x: x[1], default=(None, None))
    return max_shortest_index,min_longest_index

def find_min_editdistance(reference,responses):
    minindex=None
    mindistance=1000000
    for i in range(len(responses)):
        temp_distance=distance(reference,responses[i][1])
        # print(temp_distance)
        if temp_distance<mindistance:
            minindex=i
            mindistance=temp_distance
    return minindex,mindistance

def get_distance(reference,response):
    return distance(reference,response)

def find_max_editdistance(reference,responses):
    maxindex=None
    maxdistance=0
    for i in range(len(responses)):
        temp_distance=distance(reference,responses[i][1])
        if temp_distance>maxdistance:
            maxindex=i
            maxdistance=temp_distance
    return maxindex,maxdistance

def get_last_id(file,key):
    if os.path.exists(file):
        with open(file,mode='r', encoding='utf-8') as infile:
            lines=infile.readlines()
            if len(lines)>0:
                lastmessage=json.loads(lines[-1])
                return lastmessage.get(key),False
            else:
                return None,True
    else:
        return None,True

def get_model_prompt(instruction,opt):
    if isinstance(instruction,str):
        if opt.model_name=="Llama3":
            prompts = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"+instruction+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif opt.model_name=="Qwen2":
            prompts = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+instruction+"<|im_end|>\n<|im_start|>assistant\n"
        elif opt.model_name=="Qwen2_5":
            prompts = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+instruction+"<|im_end|>\n<|im_start|>assistant\n"
        elif opt.model_name=="Qwen2_5_math":
            prompts = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"+instruction+"<|im_end|>\n<|im_start|>assistant\n"
        elif opt.model_name=="mathstral":
            prompts = instruction+"\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        else:
            raise ValueError
    elif isinstance(instruction,dict):
        assert 'system' in instruction.keys()
        assert 'user' in instruction.keys()
        if opt.model_name=="Llama3":
            prompts = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"+instruction["system"]+"<|eot_id|><|start_header_id|>user<|end_header_id|>"+instruction["user"]+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif opt.model_name=="Qwen2":
            prompts = "<|im_start|>system\n"+instruction["system"]+"<|im_end|>\n<|im_start|>user\n"+instruction["user"]+"<|im_end|>\n<|im_start|>assistant\n"
        elif opt.model_name=="Qwen2_5":
            prompts = "<|im_start|>system\n"+instruction["system"]+"<|im_end|>\n<|im_start|>user\n"+instruction["user"]+"<|im_end|>\n<|im_start|>assistant\n"
        elif opt.model_name=="Qwen2_5_math":
            prompts = "<|im_start|>system\n"+instruction["system"]+"<|im_end|>\n<|im_start|>user\n"+instruction["user"]+"<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError
    else:
        raise TypeError
    return prompts


def convert_json2csv(json_path):
    output_csv_path=os.path.splitext(json_path)[0]+".csv"
    json_file=load_jsonl(json_path)
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        title=json_file.keys()
        writer.writerow(title)
        first_row = json_file.values()
        writer.writerow(first_row)
        
def get_sorted_checkpoints(path):
    directory = path
    entries = os.listdir(directory)
    if len(entries)<5:
        return None
    checkpoints = [entry for entry in entries if entry.startswith('checkpoint-') and entry[11:].isdigit()]
    
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    
    return sorted_checkpoints


def check_exists_and_isempty(filepath):
    # return true only if file exists and len(file)>0
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            if len(f.readlines())>0:
                return True
            else:
                return False
    else:
        return False
    
def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    if pattern.search(text):
        return True
    else:
        return False

if __name__== "__main__" :
    fire.Fire(convert_json2csv)
