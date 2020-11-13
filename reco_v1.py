import argparse
import pickle
import numpy as np
import pandas as pd

def _get_sim_article(query_id, topn):
    with open('./base_lab_cossim.pkl', mode='rb') as f1:
        cos_sim = pickle.load(f1)

    with open('./base_lab_mapping_to_idx.pkl', mode='rb') as f2:
        artid_to_idx = pickle.load(f2)
        
    with open('./base_lab_mapping_to_id.pkl', mode='rb') as f3:
        idx_to_artid = pickle.load(f3)

    query_idx = artid_to_idx[query_id]
    sim_scores = list(enumerate(cos_sim[query_idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_scores = sim_scores[1:topn+1]
    top_sim_indices = [i[0] for i in top_sim_scores]
    top_sim_ids = [idx_to_artid[i] for i in top_sim_indices]

    return top_sim_ids
