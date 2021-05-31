import pandas as pd
import mlflow
import mlflow.keras
import flask
import tensorflow 
import tensorflow.keras as k
from predict_result import did_the_soup_lie

model_path = "models/soup"
model = mlflow.keras.load_model(model_path)

app = flask.Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def get_payload():
    path_to_replay = "./data/test"
    params = flask.request.args

    payload = {"is_lie": did_the_soup_lie(path_to_replay, model)}

    return flask.jsonify(payload)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
