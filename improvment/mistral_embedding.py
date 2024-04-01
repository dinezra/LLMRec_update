import logging
import os
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import load_json, get_emb_dim, pca_reduce_dim
data_path = "/home/ezradin/LLMRec_update/LLMRec/data/netflix/"

tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')
model = model.to("cuda")


class Mistral_embedding():
    def __init__(self, data_path:str = None, max_length:int = 4096, num_batch:int = None) -> None:
        # load model and tokenizer

        self.embedding_dim = get_emb_dim(data_path)
        self.num_batch = num_batch
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        path_to_save = Path(data_path).parent.parent
        log_file = os.path.join(path_to_save, f'create_embedding_log_{self.num_batch}.txt')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    
    def get_embeddings(self, message):
        input_texts = message
        batch_dict = tokenizer(input_texts, padding=True, return_tensors="pt")
        model_inputs = batch_dict.to('cuda')

        outputs = model(**model_inputs)

        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        outputs = None
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        sqeeze_embedding = pca_reduce_dim(embeddings, self.embedding_dim)
        return sqeeze_embedding.mean(axis=0)


def entities_to_embbeding(data_path, movie_entities_path, num_batch):
    count = 0
    movies_entities_dict = load_json(movie_entities_path)
    file_path = data_path + f'/movies_entities_embedding_batch_{num_batch}'
    try:
        movie_entities_dict_loaded = pickle.load(open(file_path, 'rb'))
        loaded = True
        
    except:
        loaded = False
    mis_emb = Mistral_embedding(data_path=data_path, num_batch=num_batch)
    movies_entities_embedding = {}
    mis_emb.logger.info(("="*80))
    mis_emb.logger.info((f"\t\t\t\t\t\t\tStart process movies in batch {num_batch}"))
    mis_emb.logger.info(("="*80))
    for movie, entities in tqdm(movies_entities_dict.items()):
        if len(entities) == 0 :
            movies_entities_embedding[movie] = -1
            continue

        if (loaded and (movie not in movie_entities_dict_loaded.keys())) or (not loaded): 
            
            try:

                movies_entities_embedding[movie] = mis_emb.get_embeddings(entities)
            except:

                count += 1
                pickle.dump(movies_entities_embedding, open(file_path,'wb'))

                movies_entities_embedding[movie] = -1
        
            mis_emb.logger.info(f'Total movies process: {count} ')
        else:
            movies_entities_embedding[movie] = movie_entities_dict_loaded[movie]
    mis_emb.logger.info(f"\nSave dict successfully on {data_path + 'movies_entities_embedding'}!!\n")

    
def main():
    num_batch = 10
    data_path = '/home/ezradin/LLMRec_update/LLMRec/data/netflix'
    movie_entities_path = os.path.join(data_path, f'movie_entities_dict_batch_{num_batch}.json')
    entities_to_embbeding(data_path, movie_entities_path, num_batch)
    x = pickle.load(open(f'/home/ezradin/LLMRec_update/LLMRec/data/netflix/movies_entities_embedding','rb'))
    print()
if __name__ == "__main__":
    main()
    
    
# passages = [
#     "To bake a delicious chocolate cake, you'll need the following ingredients: all-purpose flour, sugar, cocoa powder, baking powder, baking soda, salt, eggs, milk, vegetable oil, and vanilla extract. Start by preheating your oven to 350°F (175°C). In a mixing bowl, combine the dry ingredients (flour, sugar, cocoa powder, baking powder, baking soda, and salt). In a separate bowl, whisk together the wet ingredients (eggs, milk, vegetable oil, and vanilla extract). Gradually add the wet mixture to the dry ingredients, stirring until well combined. Pour the batter into a greased cake pan and bake for 30-35 minutes. Let it cool before frosting with your favorite chocolate frosting. Enjoy your homemade chocolate cake!",
#     "The flu, or influenza, is an illness caused by influenza viruses. Common symptoms of the flu include a high fever, chills, cough, sore throat, runny or stuffy nose, body aches, headache, fatigue, and sometimes nausea and vomiting. These symptoms can come on suddenly and are usually more severe than the common cold. It's important to get plenty of rest, stay hydrated, and consult a healthcare professional if you suspect you have the flu. In some cases, antiviral medications can help alleviate symptoms and reduce the duration of the illness."
# ]

# m = Mistral_embedding()
# emb = m.get_embeddings(passages)
# print()
# Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'
# queries = [
#     get_detailed_instruct(task, 'How to bake a chocolate cake'),
#     get_detailed_instruct(task, 'Symptoms of the flu')
# ]
# No need to add instruction for retrieval documents





# get the embeddings

