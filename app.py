#-*- coding: utf-8 -*-

import os
from flask import Flask, flash, render_template, request, redirect, jsonify, session, send_from_directory, make_response
import datetime
import search
import reco_v1

app = Flask(__name__)
app.config["DEBUG"] = True
#App.secret_key = 'by hcclab'

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
        print('result', result[0])
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
def recommend():
    cur = request.args.get("id")
    #cur = int('1081592')
    print('current news item is...', cur)

    if (cur is not None):
        cur = int(cur)
        top_sim_ids = reco_v1._get_sim_article(cur, topn=10)
        print('top n', len(top_sim_ids))
        print('most similar id is...', top_sim_ids[0])
        #pagination = {'current':p, "total":total//10, "max":total}

    else:
        print("No query id was given...!")
        return make_response(render_template('index.html'))

    #if total == 0: 
    #    return make_response(render_template('no_res.html'))
        #result, total = search.issue(s,page=p,product=product,sort=sort,ctgr=rctgr,op='or')
        
    response = make_response(render_template('reco_v1.html', query_id=cur, result=top_sim_ids))
    return response

#@app.route('/result', methods=['POST'])
#def result():
#    query = request.form.get('query_')
#    total = request.form.get('total_')
#    result = request.form.get('result_')
    #result = result.split(',')

#    response = make_response(render_template('result.html', query=query, result=result, total=total))
#    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#    response.headers['Pragma'] = 'no-cache'
#    print('response made', query)
    #return render_template('page2.html', nut=nut,name=name,pct=pct)

#    return response

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