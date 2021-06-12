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
        print(type(full_input))
        print(full_input.shape)
        test_input = full_input[:,:ix_timestamp+1,:]

    pred = model.predict(test_input, verbose=False)

    rounded_result = int(pred.round()[0][0])

    # the supply lied if player 1 loses and the model predict it would have won
    if (participants[1] == "Loss") and (rounded_result == 1):
        return True
    else:
        return False
