import json
import os
import pickle
import re
import pandas as pd
from textblob import TextBlob
from torch import Tensor
import torch
import numpy as np
from sklearn.decomposition import PCA
import nltk
# nltk.download('all')
device = 'cuda'


def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
    print(f"\nSave dict successfully on {file_path}!!\n")


def read_adjacency_list(file_path):
    adjacency_list_dict = {}
    train_mat = pickle.load(open(file_path + 'train_mat','rb'))
    for index in range(train_mat.shape[0]):
        data_x, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y
    return adjacency_list_dict

def construct_prompting(task, content):
    
    if task == 1:
        message = [{"role": "user",
         "content": f"Now you are a film journalist, write your summary for the movie - {content}"}]
        
    else:
        message = [{"role": "user",
         "content": f"""
         Return back the 15 most significant entities (Do not include the name of the movie, and only max 3 words for entity) in the following movie summarize,
         the output need to be in Json format- 
         [
            1: enrirty1,
            2: entity2,
            .
            .
            .    
         ]
         {content}"""}]
        
    return message



def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def output_parser(text):
    text = text.replace('\n', '')
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s,]', '', text)


    # Tokenize and tag with Part-of-Speech
    blob = TextBlob(text)
    
    nouns = []
    current_name = ""
    for (word, pos) in blob.tags:
        if pos == 'NNP':  # Proper noun
            current_name += " " + word
        elif current_name:
            nouns.append(current_name.strip())
            current_name = ""
    if current_name:
        nouns.append(current_name.strip())

    return nouns


def get_emb_dim(data_path):
    augmented_total_embed_dict_path = os.path.join(data_path, 'augmented_total_embed_dict')
    augmented_total_embed_dict_LLMRec = pickle.load(open(augmented_total_embed_dict_path,'rb')) 
    return len(augmented_total_embed_dict_LLMRec['year'][0])


def pca_reduce_dim(list_of_tensors, dim):
    
    # pca_results = []
    # for tensor in list_of_tensors:
    #     tensor_np = tensor.detach().numpy() 
    #     if dim is not None:
    #         tensor_np = tensor_np.transpose(dim, 0).reshape(tensor_np.shape[dim], -1)
    #     pca = PCA(n_components=min(tensor_np.shape[0], tensor_np.shape[1]))
    #     pca.fit(tensor_np)
    #     tensor_pca = pca.transform(tensor_np)
    #     tensor_pca = torch.from_numpy(tensor_pca)
    #     pca_results.append(tensor_pca)
    return list_of_tensors[:,:1500]



def combine_entities_emb(data_path):
    
    combined_data = []
    for i in range(1,11):
        movies_entities_embedding_batch_path = data_path + f'movies_entities_embedding_batch_{i}'
        movies_entities_batch = pickle.load(open(movies_entities_embedding_batch_path, 'rb'))
        combined_data.append(movies_entities_batch)
        
    save_path = os.path.join(data_path, "movies_entities_embedding_finall")
    pickle.dump(combined_data, open(save_path, 'wb'))
    return combined_data

        
def split_and_save_batches(df, batch_size, start_idx, path_file, index_only=False):
    if not os.path.exists(path_file):
        os.makedirs(path_file)
    indexes = []
    num_batches = (len(df) - start_idx) // batch_size + 1
    for i in range(batch_size):
        batch_start = start_idx+(i) * num_batches
        batch_end = min(start_idx + (i + 1) * num_batches, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        batch_filename = os.path.join(path_file, f"item_attribute_filter_batch_{i+1}.csv")
        indexes.append((batch_start, batch_end))
        if not index_only:
            batch_df.to_csv(batch_filename, index=False)
            print(f"Batch {i} saved to {batch_filename}")
    if index_only:
          return indexes



