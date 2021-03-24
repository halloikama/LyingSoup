import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
import mlflow.keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from replay_processing.replay_preprocess import process_replays_from_local_folder

MODEL_PATH ="models/soup"

def train(X, Y, n_epochs, validation_split, batch_size):
    model = Sequential()
    model.add(LSTM(32, input_shape=(None,2), return_sequences=False, activation='relu'))
    model.add(Dense(1, activation=('sigmoid')))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X, Y, epochs=n_epochs, validation_split=validation_split, verbose=1, batch_size=batch_size)
    return model

def train_model_from_local_files(path_to_replays, save_df=False, from_json=False, save_trained_model=True):
    
    if from_json:
        train_df = pd.read_json("data/preprocessed_files.json")
    else:
        train_df,_, _ , _, _ = process_replays_from_local_folder(path_to_replays, VERBOSE=False)
        #train_df = train_df.reset_index()

        if save_df:
            print("saving processed df")
            train_df.to_json("data/preprocessed_files.json")
    

    # get categorical y-label (if player 1 wins == 1 else 0)
    Y = train_df.winner
    Y = (to_categorical(Y)[:,2]).astype(int)
    Y = 1 - Y
    train_y = np.array([[y] for y in Y])

    padded_X = pad_sequences(train_df.XCV, dtype='float32', padding='pre')

    trained_model = train(padded_X, train_y, n_epochs=5, validation_split=0.1, batch_size=10)

    if save_trained_model:
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        mlflow.keras.save_model(trained_model, MODEL_PATH)

if __name__ == "__main__":
    train_model_from_local_files("data/all_data", save_df=0, from_json=0, save_trained_model=True)

