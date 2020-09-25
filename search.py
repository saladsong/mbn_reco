from elasticsearch import Elasticsearch
import json

def issue(s='콜라', page=1):
    if page == None or "":
        doc_from, doc_to = 0, 10
    else:
        doc_from, doc_to = (int(page)-1)*10,(int(page))*10

    es_client = Elasticsearch("localhost:9200",timeout=30)

    with open('./body.json',mode='rt') as f:
        query = json.load(f)
        f.close()

    #body['query']['bool']['must'][0]['bool']['should'][1]['match_phrase']['Issue Details'].update({"query":s})
    query['query']['multi_match'].update({"query": s})
    query.update({"from":doc_from})

    #if sort == 'latest':
    #    query.update({"sort":[{"Registered date": {"order": "desc"}}]})

    #response = es_client.search(index="issue-v0.1.4",body=body)
    response = es_client.search(index="mbn_econ", body=query)
    hits = response['hits']['hits']
    total = response['hits']['total']['value']

    return hits, total