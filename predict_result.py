import mlflow.keras
import numpy as np
from replay_processing.replay_preprocess import process_replays_from_local_folder

def predict(test_input, model_path = "models/soup"):
    model = mlflow.keras.load_model(model_path)
    test_output = model.predict(test_input, verbose=False)
    return test_output

def did_the_soup_lie(prediction, players):
    rounded_result = int(prediction.round()[0][0])
    if (players[1] == 'Loss') and (rounded_result == 1):
        return True
    else:
        return False

def main():
    full_df, participants, data_dict_full, parse, replay_stats = process_replays_from_local_folder('./data/test', VERBOSE=False)
    test_input = np.array(parse.get('seq'))
    output = predict(test_input)

    #print(test_input[0][:5])

    if did_the_soup_lie(output,participants) == True:
        print("The soup DOES lie")
    else:
        print("The soup didn't lie")

    

if __name__ == "__main__":
    main()