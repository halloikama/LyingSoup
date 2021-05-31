import mlflow.keras
import numpy as np
from replay_processing.replay_preprocess import process_replays_from_local_folder


def did_the_soup_lie(path_to_replay, model):
    (
        full_df,
        participants,
        data_dict_full,
        parse,
        replay_stats,
    ) = process_replays_from_local_folder(path_to_replay, VERBOSE=False)
    test_input = np.array(parse.get("seq"))
    pred = model.predict(test_input, verbose=False)

    rounded_result = int(pred.round()[0][0])

    # the supply lied if player 1 loses and the model predict it would have won
    if (participants[1] == "Loss") and (rounded_result == 1):
        return True
    else:
        return False


def main():
    if did_the_soup_lie("./data/test") == True:
        print("The soup DOES lie")
    else:
        print("The soup didn't lie")


if __name__ == "__main__":
    main()
