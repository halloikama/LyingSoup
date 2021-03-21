import numpy as np
import pandas as pd
import os
import shutil
import mlflow.keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from replay_processing.replay_preprocess import process_replays_from_local_folder

model_path ="models/soup"

def train(X, Y, n_epochs, validation_split, batch_size):
    model = Sequential()
    model.add(LSTM(32, input_shape=(None,5), return_sequences=True, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    history = model.fit(X, Y, epochs=n_epochs, validation_split=validation_split, verbose=1, batch_size=batch_size)

    return model

def train_model_from_local_files(path_to_replays, save_df=False, from_csv=False, save_trained_model=True):
    
    if from_csv:
        train_df = pd.read_csv("data/preprocessed_files.csv", sep=';')
    else:
        train_df = process_replays_from_local_folder(path_to_replays, VERBOSE=True)
        train_df = train_df.reset_index()

        if save_df:
            train_df.to_json("data/preprocessed_files.json")
    

    # get categorical y-label (if player 1 wins == 1 else 0)
    Y = train_df.winner.to_numpy()
    Y = (to_categorical(Y)[:,2]).astype(int)
    Y = 1 - Y

    X = train_df.seq.to_numpy()
    X = pad_sequences(X)

    trained_model = train(X, Y, n_epochs=10, validation_split=0.2, batch_size=10)

    if save_trained_model:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        mlflow.keras.save_model(trained_model, model_path)


def predict(test_input):
    model = mlflow.keras.load_model(model_path)
    test_output = model.predict(test_input, verbose=False)
    return test_output

if __name__ == "__main__":
    train_model_from_local_files("data/all_data", save_df=True, from_csv=False)

