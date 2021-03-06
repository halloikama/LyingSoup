import pandas as pd
import mlflow
import mlflow.keras
import flask
import os
from flask import render_template, request, redirect, url_for, abort
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
    return payload


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

            if payload['is_lie'] == True:
                return render_template('result_lie.html',
                                        p1=payload['participants'][0], 
                                        p2 = payload['participants'][2],
                                        winner = payload['winner'],
                                        prob = payload['prob_of_winning'])
            else:
                return render_template('result.html',
                                         p1=payload['participants'][0], 
                                         p2 = payload['participants'][2],
                                         winner = payload['winner'],
                                         prob = payload['prob_of_winning'])

    return render_template('upload.html')

