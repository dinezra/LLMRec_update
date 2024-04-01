from pathlib import Path
import sys
from time import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
# from mistral_embedding import Mistral_embedding
from utils import *
import logging
import os
import torch.multiprocessing as mp
from functools import partial
torch.cuda.empty_cache()
device = "cuda" # the device to load the model onto


class World_knowledge():
    
    def __init__(self, data_path:str = None, toy_item_attribute_name:str = None, batch_num:str = None):
        """
                Parameters:
        - data_path (str): Path to the data directory.
        - toy_item_attribute_name (str): Name of the CSV file containing toy item attributes.
        """
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.toy_item_attribute = pd.read_csv(data_path +  toy_item_attribute_name + f'_batch_{batch_num}.csv', names=['id','year', 'title'])
        self.adjacency_list_dict = read_adjacency_list(data_path)
        self.data_path = data_path
        self.model.to(device)
        self.batch_num = batch_num
        self.logger = self._setup_logger()
        try:
            self.mis_emb = Mistral_embedding()
        except:
            self.mis_emb = None
    
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        path_to_save = Path(self.data_path).parent.parent
        log_file = os.path.join(path_to_save, f'create_wk_log_{self.batch_num}.txt')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger



    def process_movie(self, movie, movie_entities_dict_loaded):
        
        if movie not in movie_entities_dict_loaded.keys():
            prompt_task_1 = construct_prompting(task=1, content=movie)
            summarize_movie = self.mistral_generate(prompt_task_1)
            prompt_task_2 = construct_prompting(task=2, content=summarize_movie)
            entities_output = self.mistral_generate(prompt_task_2)
            try:
                entities = output_parser(entities_output)
                return movie, entities
            except:
                return movie, entities_output
        else:
            return movie, movie_entities_dict_loaded[movie]
            
    def mistral_generate(self, message):

        encodeds = self.tokenizer.apply_chat_template(message, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = self.model.generate(model_inputs,
                                            max_new_tokens=150,
                                            temperature=0)

        decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])

        return decoded[0]

    
    def process_movies_parallel(self) -> None :
        """
        give to llm 2 task: 
            1. summarize movie 
            2. generate semntic role labeling by summarize 
        
        """
        file_path = os.path.join(self.data_path, f'movie_entities_dict_batch_{self.batch_num}.json')
        movie_entities_dict = {}
        count = 0
        
        try:
            movie_entities_dict_loaded = load_json(file_path)
            loaded = True
            
        except:
            loaded = False
            

        # loaded -> i want to check if the movie exist
        # not loaded -> run regular        
        for movie in tqdm(self.toy_item_attribute['title']):
            if (loaded and (movie not in movie_entities_dict_loaded.keys())) or (not loaded): 
                count += 1
                start = time()
                prompt_task_1 = construct_prompting(task=1,
                                                    content=movie)
                
                summarize_movie = self.mistral_generate(prompt_task_1)
                prompt_task_2 = construct_prompting(task=2,
                                                    content=summarize_movie)
                
                entities_output = self.mistral_generate(prompt_task_2)
                try:
                    entities = output_parser(entities_output)
                    movie_entities_dict[movie] = entities
                        
                except:
                    movie_entities_dict[movie] = entities_output
                end_time = time()
                self.logger.info(f'Total movies process: {count} ')
                
                save_json(movie_entities_dict, file_path)
            
            else:
                movie_entities_dict[movie] = movie_entities_dict_loaded[movie]
        
        # file_path = os.path.join(self.data_path, 'movie_entities_dict.json')
        # movie_entities_dict_loaded = load_json(file_path)
        # movie_entities_dict = {}


        # num_processes = 8

        # # Create a partial function to pass additional arguments to process_movie
        
        # partial_process_movie = partial(self.process_movie, movie_entities_dict_loaded=movie_entities_dict_loaded)
        # start = time()
        # with mp.Pool(processes=num_processes) as pool:
        #     # Map the process_movie function to each movie in parallel
        #     results = pool.map(partial_process_movie, self.toy_item_attribute['title'][2000:2064])
            
            
        # total_time_seconds = time() - start
        # total_time_minutes = total_time_seconds / 60

        # print("="*150)
        # print(50 * '\t' + f'Total Execution Time: {total_time_minutes:.2f} minutes')
        # print("="*150)
        
        # for movie, result in results:
        #     movie_entities_dict[movie] = result

        save_json(movie_entities_dict, file_path)

                
        


def split_df_to_batch(data_path, toy_item_attribute_name):

    file_path = os.path.join(data_path, 'movie_entities_dict.json')
    movie_entities_dict_loaded = load_json(file_path)
    start_idx = len(movie_entities_dict_loaded)
    batch_size=10

    df = pd.read_csv(os.path.join(data_path, toy_item_attribute_name+'.csv')
                     , names=['id','year', 'title'])
    split_and_save_batches(df, batch_size, start_idx, data_path)
    
    
def main(data_path, toy_item_attribute_name, split_to_batch = False):
    batch_num = 1
    if split_to_batch:
        split_df_to_batch(data_path, toy_item_attribute_name)

    wk = World_knowledge(data_path=data_path,
                         toy_item_attribute_name=toy_item_attribute_name,
                         batch_num=batch_num)
    wk.logger.info(("="*80))
    wk.logger.info((f"\t\t\t\t\t\t\tStart process movies in batch {batch_num}"))
    wk.logger.info(("="*80))
    
    wk.process_movies_parallel()
    
if __name__ == '__main__':
    data_path = "/home/ezradin/LLMRec_update/LLMRec/data/netflix/"
    toy_item_attribute_name = 'item_attribute_filter'
    main(data_path, toy_item_attribute_name, split_to_batch=False)
    





# for index in toy_item_attribute[]:
#     # # make prompting
#     re = LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, augmented_sample_dict)




