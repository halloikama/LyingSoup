import mlflow.keras
import numpy as np
import math
from replay_processing.replay_preprocess import process_replays_from_local_folder, process_replays_from_file


def did_the_soup_lie(path_to_replay, model, timestamp):
    # process replay file
    (
        full_df,
        participants,
        data_dict_full,
        parse,
        replay_stats,
    ) = process_replays_from_file(path_to_replay, VERBOSE=False)

    # get the closest frame to the timestamp in the data
    # time is 1.4x of normal time on faster game speed and there are 16 frames per second
    # frame at time = seconds * 1.4 * 16.
    corrected_time_frame = timestamp*1.4*16
    closest_frame_at_time =  int(round(corrected_time_frame/160.0)) * 160

    frames_in_data = full_df['frames'][0].tolist()
    full_input = np.array(parse.get("seq"))

    if timestamp < 0 or closest_frame_at_time > max(frames_in_data):
        test_input = full_input
    elif timestamp == 0:
        test_input = full_input[0][:1]
    else:
        ix_timestamp = frames_in_data.index(closest_frame_at_time)
        test_input = full_input[:,:ix_timestamp+1,:]

    pred = model.predict(test_input, verbose=False)

    if participants[1] == 'Win':
        winner = participants[0]
        prob_of_win = pred[0][0]
    else:
        winner = participants[2]
        prob_of_win = 1-pred[0][0]

    results_dict = {'participants': [str(p) for p in participants],
                    'winner': str(winner),
                    'is_lie': False,
                    'prob_of_winning': float(prob_of_win*100),
                    'time': int(closest_frame_at_time),
                    }

    # the supply lied if player 1 loses and the model predict it would have won
    if prob_of_win < 0.5:
        results_dict['is_lie'] = True

    return results_dict