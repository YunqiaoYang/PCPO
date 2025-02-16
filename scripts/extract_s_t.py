import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from utilis import save_jsonl,load_jsonl
from tqdm import tqdm
import threading
from queue import Queue
import fire

class Extract_s_t():
    def __init__(self,scored_s_t_path,output_dir,key='s_t_values_weighted',sample_num=None,sample_method='downward',threshold=None,reverse=False,if_visualize=True):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.scored_s_t_path=scored_s_t_path
        self.sample_method = sample_method
        assert sample_method in ['upward','downward','random','threshold','threshold_lower']
        self.sample_num=sample_num
        self.dir_path = output_dir
        self.select_key = key
        self.if_visualize=if_visualize
        self.reverse = reverse

        if isinstance(threshold,list):
            self.thre = None
            self.down_thre = threshold[0]
            self.up_thre = threshold[1]
            assert self.up_thre>self.down_thre
        else:
            self.thre=threshold
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        if self.reverse:
            self.extract_path =  os.path.join(self.dir_path,f'extract_min_{self.select_key}.jsonl')
            reverse_flag="_reverse"
        else:
            self.extract_path =  os.path.join(self.dir_path,f'extract_max_{self.select_key}.jsonl')
            reverse_flag=""
        
        if self.thre:
            self.sampled_path = os.path.join(self.dir_path,f'{self.sample_method}{reverse_flag}_sampled_{self.sample_num}_thre_{self.thre}.jsonl')
        elif self.down_thre and self.up_thre:
            self.sampled_path = os.path.join(self.dir_path,f'{self.sample_method}{reverse_flag}_sampled_{self.sample_num}_thre_{self.down_thre}_{self.up_thre}.jsonl')
        else:
            self.sampled_path = os.path.join(self.dir_path,f'{self.sample_method}{reverse_flag}_sampled_{self.sample_num}.jsonl')


    def extract(self):
        
        data = load_jsonl(self.scored_s_t_path)

        df = pd.DataFrame(data)
        outputs=[]
        if self.select_key in df.columns:

            max_weighted_rows = df.loc[df.groupby(['idx', 'rejected'])[self.select_key].idxmax()]
            max_weighted_rows.drop(columns=['index_mapping','s_t_values'], inplace=True)
            for _, row in max_weighted_rows.iterrows():
                outputs.append(row.to_dict())
            save_jsonl(outputs,self.extract_path)
            print(f"Data extracted and saved to {self.extract_path}")
        else:
            print(f"The column {self.select_key} does not exist in the data.")
            
    
    def sample(self):
        df = pd.read_json(self.extract_path, lines=True)
        if self.sample_method =='upward':
            if self.thre:
                df = df[df[self.select_key] >= self.thre]
            df_sorted = df.sort_values(by=self.select_key, ascending=True)
            sampled_df = df_sorted.head(self.sample_num)
            # sampled_df = sampled_df.sample(frac=1, random_state=1)
        elif self.sample_method =='downward':
            if self.thre:
                df = df[df[self.select_key] <= self.thre]
            df_sorted = df.sort_values(by=self.select_key, ascending=False)
            sampled_df = df_sorted.head(self.sample_num)
            # sampled_df = sampled_df.sample(frac=1, random_state=1)
        elif self.sample_method =='threshold':
            df = df[df[self.select_key] <= self.up_thre]
            df = df[df[self.select_key] >= self.down_thre]
            sampled_df = df.sample(n=self.sample_num, random_state=1)
            sampled_df = sampled_df.sort_values(by=self.select_key, ascending=False)
        elif self.sample_method =='threshold_lower':
            df_keep = df[df[self.select_key] >= self.up_thre]
            df = df[df[self.select_key] <= self.up_thre]
            df = df[df[self.select_key] >= self.down_thre]
            sampled_df = df.sample(n=self.sample_num - df_keep.shape[0], random_state=1)
            df_merged = pd.concat([df_keep, sampled_df], ignore_index=True)
            sampled_df = df_merged.sort_values(by=self.select_key, ascending=False)
        elif self.sample_method =='random':
            sampled_df = df.sample(n=self.sample_num, random_state=1)
        sampled_df.to_json(self.sampled_path, lines=True, orient='records')
    
    def run(self):
        if not os.path.exists(self.extract_path):
            if self.reverse:
                self.extract_reverse()
            else:
                self.extract()
        if not os.path.exists(self.sampled_path) and self.sample_num:
            self.sample()
        if self.if_visualize:
            self.vis_distance_vs_reward()

    
if __name__=="__main__":
    fire.Fire(Extract_s_t)