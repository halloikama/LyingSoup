import pandas as pd
import mlflow
import mlflow.keras
import flask
import os
import json
import ast
from flask import render_template, request, redirect, url_for, abort, jsonify
from werkzeug.utils import secure_filename
from predict_result import did_the_soup_lie

import sc2reader
from sc2reader.engine.plugins import APMTracker, ContextLoader, SelectionTracker

model_path = "models/soup"
model = mlflow.keras.load_model(model_path)

app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.SC2Replay']
app.config['UPLOAD_PATH'] = 'uploads'

def get_result(replay_file, time):
    payload = did_the_soup_lie(replay_file, model, time)
    #payload = flask.jsonify(payload)
    return payload

@app.route('/result', methods=['GET'])
def show_results():
    payload = request.args['payload']
    results_dict = ast.literal_eval(payload)
    return render_template('result.html',
                                        is_lie = results_dict['is_lie'],
                                        p1=results_dict['participants'][0], 
                                        p2 = results_dict['participants'][2],
                                        winner = results_dict['winner'],
                                        prob = results_dict['prob_of_winning'],
                                        time = results_dict['time'])
    
@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST': 
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        # check if file was uploaded
        if filename != '':
            # check if file has .SC2Replay extension
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return "Not a replay file", 400

            # get the timestamp in seconds
            time_min = request.form.get('minutes')
            time_sec = request.form.get('seconds')

            if time_min != '':
                time_min = int(time_min)
            elif time_sec != '':
                time_min = int(0)
            else:
                time_min = int(-1)

            if time_sec != '':
                time_sec = int(time_sec)
            elif time_min != '':
                time_sec = int(0)     
            else:
                time_sec = int(-1)           

            timestamp = time_min*60+time_sec

            payload = get_result(uploaded_file, timestamp)

            return redirect(url_for('show_results', payload=payload, name=filename))

            '''
            return render_template('result.html',
                                        is_lie = payload['is_lie'],
                                        p1=payload['participants'][0], 
                                        p2 = payload['participants'][2],
                                        winner = payload['winner'],
                                        prob = payload['prob_of_winning'])
            '''
    return render_template('upload.html')



