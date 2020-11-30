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


def _get_sim_article(query_id, topn, model):
    with open('./adv_' + str(model) + '_cossim.pkl', mode='rb') as f0:
        cos_sim = pickle.load(f0)   

    with open('./base_datelist_l.pkl', mode='rb') as f1:
        date_list = pickle.load(f1)

    with open('./base_lab_mapping_to_idx.pkl', mode='rb') as f2:
        artid_to_idx = pickle.load(f2)

    with open('./base_lab_mapping_to_id.pkl', mode='rb') as f3:
        idx_to_artid = pickle.load(f3)

    query_idx = artid_to_idx[query_id]

    sim_scores = list(enumerate(cos_sim[query_idx]))    ## visual feature (label) 기반 relevancy score
    date_diff_n = calc_datediff(query_idx, date_list)
    new_sim_scores = update_simsco(sim_scores, date_diff_n)
    
    new_sim_scores = sorted(new_sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_scores = new_sim_scores[0:topn+1]
    top_sim_indices = [i[0] for i in top_sim_scores]
    top_sim_ids = [idx_to_artid[i] for i in top_sim_indices]
    
    ## in case of original item being recommended
    if query_id in top_sim_ids:
        q_idx = top_sim_ids.index(query_id)
        top_sim_ids.pop(q_idx)

    return top_sim_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--reco_model', type=str, default='tfidf', help='reco model among "tfidf(v21), w2v(v22), lda(v23)"')
    parser.add_argument('--query_id', type=int, default=1167105)
    parser.add_argument('--topn', type=int, default=15)
    args = parser.parse_args()

    # retreive most similar (Top-N) item ids 
    top_sim_ids = _get_sim_article(query_id=args.query_id, topn=args.topn, model=args.reco_model)
    print('original query id is: ', args.query_id)
    print('most similar one is: ', top_sim_ids[0])
    print('other similars..', top_sim_ids)