#-*- coding: utf-8 -*-

import os
from flask import Flask, flash, render_template, request, redirect, jsonify, session, send_from_directory, make_response
import pandas as pd
import datetime
import search
import reco_v1
import reco_v2

app = Flask(__name__)
app.config["DEBUG"] = True

#meta_path = "./data_econ.csv"
#meta_data = pd.read_csv(meta_path, encoding='utf-8')


def save_log(log):
	with open('./log/time_eventlog', 'a') as f:
		now = datetime.datetime.now()
		logWtime = "["+now.strftime('%Y-%m-%d %H:%M:%S')+"] "+log+"\n"
		f.write(logWtime)

@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    
    return response

@app.route('/ret', methods=['POST'])
def ret():
    s = request.form["search"]
    #s = request.form.get('search')
    print('query is...', s)

    if (s is not None):
        result, total = search.issue(s)
        print('total', total)
        if len(result) > 0:
            print('result', result[0])
        else:
            print('result is none')
        #pagination = {'current':p, "total":total//10, "max":total}

    else:
        print("No query was given...!")
        return make_response(render_template('index.html'))

    #if total == 0: 
    #    return make_response(render_template('no_res.html'))
        #result, total = search.issue(s,page=p,product=product,sort=sort,ctgr=rctgr,op='or')
        
    response = make_response(render_template('result.html', query=s, result=result, total=total))
    return response


@app.route('/reco_v1', methods=['GET'])
def recommend_v1():
    cur = request.args.get("id")
    print('current news item is...', cur)

    if (cur is not None):
        cur = int(cur)
        top_sim_ids_v1 = reco_v1._get_sim_article(cur, topn=10)
        #cur_meta = meta_data[meta_data['ART_ID'] == [cur]]
        #print('metadata of current doc,' , cur_meta)
        print('top n', len(top_sim_ids_v1))
        print('most similar id (v1) is...', top_sim_ids_v1)
        #pagination = {'current':p, "total":total//10, "max":total}

    else:
        print("No query id was given...!")
        return make_response(render_template('index.html'))

    #if total == 0: 
    #    return make_response(render_template('no_res.html'))
        #result, total = search.issue(s,page=p,product=product,sort=sort,ctgr=rctgr,op='or')
        
    response = make_response(render_template('reco_base.html', query_id=cur, result=top_sim_ids_v1))
    return response


@app.route('/reco_v2', methods=['GET'])
def recommend_v2():
    cur = request.args.get("id")
    print('current news item is...', cur)

    if (cur is not None):
        cur = int(cur)
        top_sim_ids_v2 = reco_v2._get_sim_article(cur, topn=10)
        print('top n', len(top_sim_ids_v2))
        print('most similar id (v2) is...', top_sim_ids_v2)
        #pagination = {'current':p, "total":total//10, "max":total}

    else:
        print("No query id was given...!")
        return make_response(render_template('index.html'))

    #if total == 0: 
    #    return make_response(render_template('no_res.html'))
        #result, total = search.issue(s,page=p,product=product,sort=sort,ctgr=rctgr,op='or')
        
    response = make_response(render_template('reco_base.html', query_id=cur, result=top_sim_ids_v2))
    return response


@app.route('/static/js/reco.js')
def add_js_header1():
	response = send_from_directory(app.static_folder, 'js/reco.js')
	response.headers['X-XSS-Protection'] = "1"
	return response


#@app.route('/issueView', methods=['GET'])
#def issueView():
#    issueId = request.args['issueId']
#    return redirect('https://ims.tmaxsoft.com/tody/ims/issue/issueView.do?issueId='+issueId)

if __name__ == '__main__': 
    app.run(host='0.0.0.0',port='8888')