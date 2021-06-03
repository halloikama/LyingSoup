import numpy as np
import pandas as pd
import os
import sc2reader
from sc2reader.engine.plugins import APMTracker, ContextLoader, SelectionTracker
from sc2reader.events import (
    PlayerStatsEvent,
    UnitBornEvent,
    UnitDiedEvent,
    UnitDoneEvent,
    UnitTypeChangeEvent,
    UpgradeCompleteEvent,
)

from replay_processing.parse_replay import ReplayData


def get_stats(replay_object):
    diff_values = [] 

    p1_minerals = replay_object.as_dict().get('stats').get(1).get('mineral_collection_rate')
    p2_minerals = replay_object.as_dict().get('stats').get(2).get('mineral_collection_rate')

    p1_gas = replay_object.as_dict().get('stats').get(1).get('vespene_collection_rate')
    p2_gas = replay_object.as_dict().get('stats').get(2).get('vespene_collection_rate')

    p1_supply = replay_object.as_dict().get('stats').get(1).get('supply_consumed')
    p2_supply = replay_object.as_dict().get('stats').get(2).get('supply_consumed')

    p1_mineralarmy = replay_object.as_dict().get('stats').get(1).get('mineral_value_current_army')
    p2_mineralarmy = replay_object.as_dict().get('stats').get(2).get('mineral_value_current_army')

    p1_gasarmy = replay_object.as_dict().get('stats').get(1).get('vespene_value_current_army')
    p2_gasarmy = replay_object.as_dict().get('stats').get(2).get('vespene_value_current_army')

    p1_wsratio = replay_object.as_dict().get('stats').get(1).get('worker_supply_ratio')
    p2_wsratio = replay_object.as_dict().get('stats').get(2).get('worker_supply_ratio')

    if len(p1_minerals) != len(p2_minerals):
        diff = len(p1_minerals) - len(p2_minerals)
        if diff < 0:
            p2_minerals = p2_minerals[:diff]
            p2_gas = p2_gas[:diff]
            p2_supply = p2_supply[:diff]
            p2_mineralarmy =  p2_mineralarmy[:diff]
            p2_gasarmy =  p2_gasarmy[:diff]
            p2_wsratio = p2_wsratio[:diff]
        else:
            p1_minerals = p1_minerals[:-diff]
            p1_gas = p1_gas[:-diff]
            p1_supply = p1_supply[:-diff]
            p1_mineralarmy =  p1_mineralarmy[:-diff]
            p1_gasarmy =  p1_gasarmy[:-diff]
            p1_wsratio = p1_wsratio[:-diff]
    
    d_minerals = np.array(list(zip(*p1_minerals))[1]) - np.array(list(zip(*p2_minerals))[1])
    d_gas = np.array(list(zip(*p1_gas))[1]) - np.array(list(zip(*p2_gas))[1])
    d_supply = np.array(list(zip(*p1_supply))[1]) - np.array(list(zip(*p2_supply))[1])
    d_mineralarmy = np.array(list(zip(*p1_mineralarmy))[1]) - np.array(list(zip(*p2_mineralarmy))[1])
    d_gasarmy = np.array(list(zip(*p1_gasarmy))[1]) - np.array(list(zip(*p2_gasarmy))[1])
    d_wsratio = np.array(list(zip(*p1_wsratio))[1]) - np.array(list(zip(*p2_wsratio))[1])
    

    zipper = zip(d_supply, d_wsratio)
    diff_values_zipped = np.array(list(zipper))
    data_dict_full = {"d_minerals":d_minerals.tolist(), "d_gas":d_gas.tolist(),"d_supply":d_supply.tolist(), "d_mineralarmy":d_mineralarmy.tolist(), 
                      "d_gasarmy":d_gasarmy.tolist(),"d_wsratio":d_wsratio.tolist(), "XCV": diff_values_zipped, "frames": np.array(list(zip(*p1_minerals))[0])}
    

    return diff_values_zipped, data_dict_full
    
def parse_input(replay_path, VERBOSE=False):
    
    try:
        replay_file = replay_path
    except NameError:
        print('\n'
              'SETUP ERROR: Please follow the directions to add a .SC2Replay file and use\n'
              '             "Insert to code" to set the streaming_body_1 variable to the resulting bytes.\n'
              '             You may need to rename the data_* variable.')
        raise

    replay = sc2reader.load_replay(
        replay_file,
        engine=sc2reader.engine.GameEngine(plugins=[ContextLoader(), APMTracker(), SelectionTracker()]))
    
    participants = []
    for plays in replay.players:
        participants.append(plays)
        participants.append(plays.result)
    
    if VERBOSE:
        print("Replay successfully loaded.")
    
        for players in replay.players:
            print(players.result, players )
    
    
    is_AI = False
    for players in replay.players:
        if isinstance(players, (sc2reader.objects.Computer)):
            is_AI = True
            if VERBOSE:
                print('found an A.I.\n')
                
        
    if (len(replay.players) != 2) or (is_AI == True):
        if VERBOSE:
            if (is_AI == True):
                print('this is an A.I. game')
            else:
                print('this is not a 1v1 game')
        return None 
    else:
        replay_object = ReplayData.parse_replay(replay=replay)
    
        try:
            y_data = list(replay_object.as_dict().get('winners'))[0][0]
        except: y_data = 3
            

        x_data, data_dict_full = get_stats(replay_object)
        data_dict_full['winner'] = y_data

        data_dict = {'seq':[x_data], 'label':[y_data]}

        return data_dict, participants, data_dict_full, replay_object.as_dict().get('stats')


def process_replays_from_local_folder(path_to_replays, VERBOSE=False):
    """

    """
    print("started processing replay file from local folder")
    full_df = pd.DataFrame()
    with os.scandir(path_to_replays) as dirs:
        for entry in dirs:
            try:
                parse, participants, data_dict_full, replay_stats = parse_input(f"{path_to_replays}/{entry.name}", VERBOSE=VERBOSE)
                full_df = full_df.append(data_dict_full, ignore_index=True)
            except:
                print(f'something went wrong, parsing of {path_to_replays}/{entry.name} failed')
                print()
                continue

    return full_df, participants, data_dict_full, parse, replay_stats

def process_replays_from_file(uploaded_file, VERBOSE=False):
    """

    """
    print("started processing replay file from file")
    full_df = pd.DataFrame()
    with open(uploaded_file, "r") as file_object: 
        try:
            parse, participants, data_dict_full, replay_stats = parse_input(file_object, VERBOSE=VERBOSE)
            full_df = full_df.append(data_dict_full, ignore_index=True)
        except:
            print(f'something went wrong, parsing of file failed')
            print()

    return full_df, participants, data_dict_full, parse, replay_stats
