import pandas as pd
import mlflow
import mlflow.keras
import flask
import tensorflow 
import tensorflow.keras as k
import os
from flask import render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from predict_result import did_the_soup_lie

model_path = "models/soup"
model = mlflow.keras.load_model(model_path)

app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.SC2Replay']
app.config['UPLOAD_PATH'] = 'uploads'

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/api', methods=['GET','POST'])    
def get_payload():
    payload = {"is_lie": False}
    payload_json = flask.jsonify(payload)
    return payload_json

@app.route('/result/<replay_file>/<time>', methods=["GET", "POST"])
def get_result(replay_file, time):
    
    # TODO: update to stream w/o saving
    path_to_replay = "./temp/"
    print(path_to_replay)

    payload = {"is_lie": did_the_soup_lie(path_to_replay, model)}

    if payload['is_lie']:
        return render_template('result_lie.html')
    else:
        return render_template('result.html')

#@app.route('/')
#def index():
#   return render_template('upload.html')

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST': 
        uploaded_file = request.files['file']
        print(uploaded_file)
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return "Not a replay file", 400
            uploaded_file.save('./temp/' + filename)
            return redirect(url_for('get_result', replay_file=filename, time = 160))

    return render_template('upload.html')

