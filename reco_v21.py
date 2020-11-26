import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import date

def calc_datediff(query_idx, date_list):
    date_diff = []
    for cand_date in date_list:
        q_date = date.fromisoformat(date_list[query_idx])
        c_date = date.fromisoformat(cand_date)
        dist_1 = (q_date - c_date)
        dist_2 = (dist_1.days)**2
        date_diff.append(dist_2)
        
    max_ = max(date_diff)
    date_diff_n = list(round(diff / max_, 6) for diff in date_diff)

    return date_diff_n

def update_simsco(sim_scores, datediff):
    new_simsco = []
    for idx, score in sim_scores:
        new_score = score - datediff[idx]
        new_simsco.append((idx, new_score))
    
    return new_simsco


def _get_sim_article(query_id, topn):
    with open('./adv_tfidf_cossim.pkl', mode='rb') as f0:
        cos_sim = pickle.load(f0)

    with open('./base_datelist_l.pkl', mode='rb') as f1:
        date_list = pickle.load(f1)

    with open('./base_lab_mapping_to_idx.pkl', mode='rb') as f2:
        artid_to_idx = pickle.load(f2)

    with open('./base_lab_mapping_to_id.pkl', mode='rb') as f3:
        idx_to_artid = pickle.load(f3)

    query_idx = artid_to_idx[query_id]

    sim_scores = list(enumerate(cos_sim[query_idx]))
    date_diff_n = calc_datediff(query_idx, date_list)
    new_sim_scores = update_simsco(sim_scores, date_diff_n)
    
    new_sim_scores = sorted(new_sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_scores = new_sim_scores[1:topn+1]
    top_sim_indices = [i[0] for i in top_sim_scores]
    top_sim_ids = [idx_to_artid[i] for i in top_sim_indices]

    return top_sim_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--query_id', type=int, default=1167105)
    parser.add_argument('--topn', type=int, default=15)
    args = parser.parse_args()

    # retreive most similar (Top-N) item ids 
    top_sim_ids = _get_sim_article(query_id=args.query_id, topn=args.topn)
    print('original query id is: ', args.query_id)
    print('most similar one is: ', top_sim_ids[0])
    print('other similars..', top_sim_ids)