import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import os
from utilis import *
from running_utilis import *
import pandas as pd
import json

import numpy as np
import tqdm
import torch
import argparse
from Levenshtein import distance
import random

import fire

class Only_Levenstein_Judge():
    def __init__(self,arg_path,iteration_dir_path,Response_input_path,Judgement_output_name):
        self.args=ArgYAMLManager.load_args_from_yaml(arg_path)

        self.dir_path=iteration_dir_path
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        
        self.input_path=Response_input_path
        self.output_path=os.path.join(self.dir_path,Judgement_output_name)
        
        self.save_mode= 'w' if 'save_mode' not in self.args.Judge_args else self.args.Judge_args['save_mode']
        
        
        self.batch_size= 200 if 'batch_size' not in self.args.Judge_args else self.args.Judge_args['batch_size']
        
        self.pair_num=5 if 'pair_num' not in self.args.Judge_args else self.args.Judge_args['pair_num']
        
        self.distance_threshold_up=800 if 'distance_threshold_up' not in self.args.Judge_args else self.args.Judge_args['distance_threshold_up']
        
        self.distance_threshold_down=200 if 'distance_threshold_down' not in self.args.Judge_args else self.args.Judge_args['distance_threshold_down']
        
        self.distance_type='min' if 'distance_type' not in self.args.Judge_args else self.args.Judge_args['distance_type']
        
        self.model_name = 'Qwen2_5' if 'model_name' not in self.args.Judge_args else self.args.Judge_args['model_name']
        
        self.based_on = 'wrong' if 'based_on' not in self.args.Judge_args else self.args.Judge_args['based_on']
        
        self.nearest = 1 if 'nearest' not in self.args.Judge_args else self.args.Judge_args['nearest']
        
    def getpair(self,task_id, base_solution_list, to_mathch_solution_list, question,based_on,nearest=1):
        k=min([self.pair_num,len(base_solution_list)])
        nearest=min([nearest,len(to_mathch_solution_list)-1])
        shuffled_index=list(range(len(base_solution_list)))
        random.shuffle(shuffled_index)
        # print(k)
        temp_outputs=[]
        origin_to_mathch_solution_list=to_mathch_solution_list.copy()
        for i in range(k):
            to_mathch_solution_list = origin_to_mathch_solution_list.copy()
            for j in range(nearest):
                if (self.distance_type == "max"):
                    index,distance=find_max_editdistance(base_solution_list[shuffled_index[i]][1],to_mathch_solution_list)
                    if index == None:
                        continue
                    if distance < self.distance_threshold_down:
                        continue
                else:
                    index,distance=find_min_editdistance(base_solution_list[shuffled_index[i]][1],to_mathch_solution_list)
                    if index == None:
                        continue

                    if distance > self.distance_threshold_up:
                            continue
                if based_on =='right':
                    output_dict = {
                        'idx': task_id,
                        'chosen':
                            {
                            'instruction':question,
                            'response':base_solution_list[shuffled_index[i]][1]
                            }
                        ,
                        'rejected':
                            {
                            'instruction':question,
                            'response':to_mathch_solution_list[index][1]
                            },
                        'distance':distance,
                        'nearest':j
                        }
                else:
                    output_dict = {
                        'idx': task_id,
                        'chosen':
                            {
                            'instruction':question,
                            'response':to_mathch_solution_list[index][1]
                            }
                        ,
                        'rejected':
                            {
                            'instruction':question,
                            'response':base_solution_list[shuffled_index[i]][1]
                            },
                        'distance':distance,
                        'nearest':j
                        }
                temp_outputs.append(output_dict)
                del to_mathch_solution_list[index]
        return temp_outputs
    
    def run(self):
        if self.save_mode == 'w':
            find_flag=True
            save_jsonl([],self.output_path,'w')
        else:
            last_id,find_flag=get_last_id(self.output_path,'idx')
            if not find_flag :
                assert last_id is not None
        with open(self.input_path, 'r',encoding='utf-8') as infile:
            total_lines=self.get_total_lines(self.input_path)
            progress_bar = tqdm.tqdm(total=total_lines)
            batch=[]
            for line_number, line in enumerate(infile, 1):
                data = json.loads(line)
                batch.append(data)
                
                if len(batch) == self.batch_size or line_number == total_lines:
                    # Process batch
                    task_ids,response_ids, questions, right_solutions, wrong_solutions= [], [], [], [], []
                    for message in batch:
                        progress_bar.update(1)
                        if find_flag is not True:
                            if last_id != message.get("idx"):
                                continue
                            find_flag=True
                            continue
                        question=message.get("question")
                        responses=message.get("code")
                        responses=[(i,response) for i, response in enumerate(responses)]
                        scores=message.get("score")

                        if True in scores and False in scores:
                            temp_right_solutions=[]
                            temp_wrong_solutions=[]
                            for (i,response),score in zip(responses,scores):
                                if score == True:
                                    temp_right_solutions.append((i,response))
                                if score == False:
                                    temp_wrong_solutions.append((i,response))
                            task_ids.append(message.get("idx"))
                            right_solutions.append(temp_right_solutions)
                            wrong_solutions.append(temp_wrong_solutions)
                            questions.append(question)


                    if len(questions)==0:
                        batch = []
                        continue
                    
                    outputs=[]
                    for task_id, right_solution_list, wrong_solution_list, question in zip(task_ids,right_solutions,wrong_solutions,questions):
                        if self.based_on =='right':
                            temp_outputs = self.getpair(task_id, right_solution_list, wrong_solution_list, question,'right',self.nearest)
                        else:
                            temp_outputs = self.getpair(task_id, wrong_solution_list, right_solution_list, question,'wrong',self.nearest)
                        outputs.extend(temp_outputs)
                    save_jsonl(outputs,self.output_path,'a')
                    print('saved')
                    batch = []
                    if line_number == total_lines:
                        break
    

def main():
    fire.Fire(Only_Levenstein_Judge)


if __name__ == "__main__":
    main()