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

def aggr_simsco(b_sim, a_sim, p=0.8):
    new_simsco = []
    for b_score, a_score in zip(b_sim, a_sim):
        _, b_sco = b_score
        _, a_sco = a_score
        new_score = p*b_sco + (1-p)*a_sco
        new_simsco.append(new_score)
        
    return new_simsco

def update_simsco(sim_scores, datediff):
    new_simsco = []
    for idx, score in enumerate(sim_scores):
        new_score = score - datediff[idx]
        new_simsco.append((idx, new_score))
    
    return new_simsco


def _get_sim_article(query_id, topn, model):
    with open('./base_tfidf_cossim.pkl', mode='rb') as f0:
        b_cos_sim = pickle.load(f0)

    with open('./adv_' + str(model) + '_cossim.pkl', mode='rb') as f1:
        a_cos_sim = pickle.load(f1)    

    with open('./base_datelist_l.pkl', mode='rb') as f2:
        date_list = pickle.load(f2)

    with open('./base_lab_mapping_to_idx.pkl', mode='rb') as f3:
        artid_to_idx = pickle.load(f3)

    with open('./base_lab_mapping_to_id.pkl', mode='rb') as f4:
        idx_to_artid = pickle.load(f4)

    query_idx = artid_to_idx[query_id]

    b_sim_scores = list(enumerate(b_cos_sim[query_idx]))   ## metadata (script) 기반 relevancy
    a_sim_scores = list(enumerate(a_cos_sim[query_idx]))   ## visual feature (label) 기반 relevancy
    aggr_scores = aggr_simsco(b_sim_scores, a_sim_scores)
    
    date_diff_n = calc_datediff(query_idx, date_list)
    new_sim_scores = update_simsco(aggr_scores, date_diff_n)
    
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
    parser.add_argument('--reco_model', type=str, default='tfidf', help='reco model among "tfidf(v31), w2v(v32), lda(v33)"')
    parser.add_argument('--query_id', type=int, default=1167105)
    parser.add_argument('--topn', type=int, default=15)
    args = parser.parse_args()

    # retreive most similar (Top-N) item ids 
    top_sim_ids = _get_sim_article(query_id=args.query_id, topn=args.topn, model=args.reco_model)
    print('original query id is: ', args.query_id)
    print('most similar one is: ', top_sim_ids[0])
    print('other similars..', top_sim_ids)