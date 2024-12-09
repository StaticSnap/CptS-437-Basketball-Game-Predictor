import pandas as pd
from google.colab import drive
import zipfile
from datetime import datetime
import shutil
import json
import random
import os
from tqdm import tqdm
import csv

# Directory for test games
test_dir = '/content/data/NBAdata-test'

# Number of games to parse
GAMES = 2000

# Randomly select 300 games
SELECTED_GAMES = 50
selected_games_indices = set(random.sample(range(1, GAMES + 1), SELECTED_GAMES))

# Define vocabularies
rosters_vocab = {'start': 1}
event_vocab = {'start': 1, 'shot': 2, 'assist': 3, 'turnover': 4,
               'rebound': 5, 'foul': 6, 'block': 7, 'substitution': 8,
               'steal': 9}
player_vocab = {'start': 1, "null": 2}
type_vocab = {'start': 1, 'null': 2, '2pt': 3, '3pt': 4, 'free throw': 5,
              'personal': 6, 'shooting': 7, 'offensive': 8,
              'loose ball': 9, 'technical': 10, 'personal take': 11,
              'flagrant-1': 12, 'away from play': 13, 'transition take': 14,
              'flagrant-2': 15, 'bonus': 16, 'steal': 17, 'violation': 18,
              'error': 19, 'rebound offensive': 20, 'rebound defensive': 21,
              'team rebound': 22}
result_vocab = {'start': 1, 'null': 2, 'made': 3, 'missed': 4, 'block': 5}
season_vocab = {'start': 1}

# Initialize IDs
rosters_id = 2
event_id = 10
player_id = 3
type_id = 23
result_id = 6
season_id = 2

vocabularies = {
    "rosters_vocab.json": rosters_vocab,
    "event_vocab.json": event_vocab,
    "player_vocab.json": player_vocab,
    "type_vocab.json": type_vocab,
    "result_vocab.json": result_vocab,
    "season_vocab.json": season_vocab,
}

# Function to save test data to individual JSON files
def save_test_data(output_dir, game_data, line):
    os.makedirs(output_dir, exist_ok=True)

    # Save data to a JSON file
    json_path = os.path.join(output_dir, f'test_game_{line}.json')
    game_df = pd.DataFrame(game_data)
    game_df.to_json(json_path, orient='records', lines=True)
    print(f"Saved test data for game {line} to {json_path}")

# Helper function to convert elapsed time into seconds
def convert_time(row):
    game_time = 0
    if int(row[13]) > 4:
        game_time = (48 + ((int(row[13]) - 5) * 5)) * 60
    else:
        game_time += ((int(row[13]) - 1) * 12 * 60)
    game_time += int(row[17][5:])
    game_time += int(row[17][2:4]) * 60

    if game_time < 1:
        game_time = 1

    return game_time

# Helper function to clear a file for deletion
def clear_file(path):
  for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

# Helper function to determine shot points
def determine_points(shot_name):
    return '3pt' if shot_name.startswith('3') else '2pt'

# Hash function for roster
def hash_roster(roster):
    if len(roster) <= 3:
        return 'start'
    prime = 31
    encoded_value = 0
    for i, num in enumerate(roster):
        encoded_value += num * (prime ** i)
    return str(encoded_value)

def encode_players(roster):
    for player in roster:
        if player not in player_vocab:
            player_vocab[player] = len(player_vocab) + 1
    roster = sorted([player_vocab[player] for player in roster])
    return roster

# Helper function to determine turnover type
def determine_turnover_type(data):
    check_vio = [
        '3-second violation', 'shot clock', '8-second violation', 'lane violation', 'offensive goaltending',
        'palming', 'backcourt', '5-second violation', 'double dribble', 'discontinue dribble', 'illegal assist',
        'jump ball violation', 'offensive foul', 'illegal screen', 'basket from below', 'punched ball',
        'too many players', 'traveling', 'kicked ball'
    ]
    check_error = ['lost ball', 'out of bounds lost ball', 'step out of bounds', 'bad pass', 'inbound']
    if data == '':
        return 'null'
    elif data in check_vio:
        return 'violation'
    elif data in check_error:
        return 'error'
    else:
        print("FLAG UNRECOGNIZED TURNOVER")
        return 'unrecognized'

def determine_foul_type(data, bonus):
  if bonus:
    return 'bonus'
  if data == 'offensive charge':
    return 'offensive'
  if data == '':
    return 'null'
  if data[-9:] == 'technical':
    return 'technical'
  return data

def determine_foul_result(data):
    if data == 'bonus':
        return 'bonus'
    elif data == 'personal' or data == 'null' or data == 'away from play':
      return 'nothing'
    elif data == 'shooting' or data == 'technical':
      return 'free throw'
    elif data == 'personal take' or data == 'flagrant-1' or data == 'transition take':
      return 'free throw op'
    elif data == 'offensive':
      return 'cop'
    elif data == 'loose ball':
      return 'op'
    elif data == 'flagrant-2':
      return 'ejection'
    else:
      raise ValueError(f"Unknown foul type: {data}")

def final_two(time):
    if time < 2800:
        valid_ranges = [
            (10 * 60, 12 * 60),
            (22 * 60, 24 * 60),
            (34 * 60, 36 * 60),
            (46 * 60, 48 * 60),
        ]
        return any(start <= time <= end for start, end in valid_ranges)
    else:
        base_time = 2800
        interval_start = base_time + 3 * 60
        interval_end = base_time + 5 * 60
        offset = time - interval_start
        return 0 <= offset % (5 * 60) < (2 * 60)

def update_vocabs(play):
    global event_id, player_id, type_id, result_id, rosters_id, season_id

    # Update season vocabulary
    season = play['season']
    if season and season not in season_vocab:
        season_vocab[season] = season_id
        season_id += 1

    # Update event vocabulary
    event = play['event']
    if event and event not in event_vocab:
        event_vocab[event] = event_id
        event_id += 1

    # Update player vocabulary
    player = play['player']
    if player and player not in player_vocab:
        player_vocab[player] = player_id
        player_id += 1

    # Update type vocabulary
    type_instance = play['type']
    if type_instance == '':
        type_instance = ''
    if type_instance and type_instance not in type_vocab:
        type_vocab[type_instance] = type_id
        type_id += 1

    # Update result vocabulary
    result = play['result']
    if result and result not in result_vocab:
        result_vocab[result] = result_id
        result_id += 1

    # Update roster vocabularies
    roster1 = hash_roster(play['roster1'])
    if roster1 and roster1 not in rosters_vocab:
        rosters_vocab[roster1] = rosters_id
        rosters_id += 1
    roster2 = hash_roster(play['roster2'])
    if roster2 and roster2 not in rosters_vocab:
        rosters_vocab[roster2] = rosters_id
        rosters_id += 1

# List to accumulate all game data
all_game_data = []
game = 0

# Directory setup
csv_files = [f for f in os.listdir('/content/data/NBAdata-dirty/NBAdata-dirty') if f.endswith('.csv')]

# Randomlines
total_files = len(csv_files)

# Output file
output_file = '/content/data/NBADATA.json'

# Clear the file at the start
open(output_file, 'w').close()

with open(output_file, 'w') as json_file:
    json_file.write('[\n')
    first_entry = True

    # Process each CSV file
    for filename in tqdm(csv_files, desc="Processing files", unit="file"):
        file_path = os.path.join('/content/data/NBAdata-dirty/NBAdata-dirty', filename)

        if game == GAMES:
            break
        game += 1

        # Extract season and playoff values
        season = filename[1:5]
        month = filename[6:7]
        if int(month) > 9:
            season = str(int(season) + 1)
        playoff = 0

        game_data_list = []

        team1 = []
        team2 = []

        team1_fouls = 0
        team2_fouls = 0

        team1_bonus = False
        team2_bonus = False

        quarter = 1

        # Open and process each CSV file
        # This part handles refactoring input data to be standardized and sepparated correctly for models.
        with open(file_path, mode='r', newline='') as file:

            csv_reader = csv.reader(file)
            next(csv_reader)

            first = True
            for row in csv_reader:

                time = convert_time(row)

                if quarter < 5:
                    if time > quarter * 60 * 12:
                        quarter += 1
                        team1_fouls = 0
                        team2_fouls = 0
                        team1_bonus = False
                        team2_bonus = False
                else:
                    if time > quarter * 60 * 5:
                        quarter += 1
                        team1_fouls = 0
                        team2_fouls = 0
                        team1_bonus = False
                        team2_bonus = False

                roster1 = encode_players([row[3], row[4], row[5], row[6], row[7]])
                roster2 = encode_players([row[8], row[9], row[10], row[11], row[12]])

                team1.extend(player for player in roster1 if player not in team1)
                team2.extend(player for player in roster2 if player not in team2)

                if first == True:
                    playoff = row[1][8]
                    if playoff == 'R':
                        playoff = 0
                    else:
                        playoff = 1

                    start_data = {
                        'gameId': game,
                        'roster1': roster1,
                        'roster2': roster2,
                        'time': 1,
                        'event': 'start',
                        'player': 'start',
                        'type': 'start',
                        'result': 'start',
                        'season': season,
                        'playoff': playoff,
                    }
                    update_vocabs(start_data)
                    game_data_list.append(start_data)
                    first = False

                # Case for assists, blocks, and normal shots
                if row[21] == 'shot':
                    if row[22] != '':
                        assist_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': 'assist',
                            'player': row[22] if row[22] != '' else 'null',
                            'type': determine_points(row[37]) if determine_points(row[37]) != '' else 'null',
                            'result': 'score',
                            'season': season,
                            'playoff': playoff
                        }

                        update_vocabs(assist_data)
                        game_data_list.append(assist_data)

                        shot_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': row[21],
                            'player': row[31] if row[31] != '' else 'null',
                            'type': determine_points(row[37]) if determine_points(row[37]) != '' else 'null',
                            'result': row[35],
                            'season': season,
                            'playoff': playoff
                        }

                        update_vocabs(shot_data)
                        game_data_list.append(shot_data)

                    elif row[25] != '':
                        shot_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': row[21],
                            'player': row[31] if row[31] != '' else 'null',
                            'type': determine_points(row[37]) if determine_points(row[37]) != '' else 'null',
                            'result': 'block',
                            'season': season,
                            'playoff': playoff
                        }

                        update_vocabs(shot_data)
                        game_data_list.append(shot_data)

                        block_data = {
                            'gameId': game,
                            'roster1': encode_players([row[3], row[4], row[5], row[6], row[7]]),
                            'roster2': encode_players([row[8], row[9], row[10], row[11], row[12]]),
                            'time': time,
                            'event': 'block',
                            'player': row[25] if row[25] != '' else 'null',
                            'type': row[31] if row[31] != '' else 'null',
                            'result': 'block',
                            'season': season,
                            'playoff': playoff
                        }

                        update_vocabs(block_data)
                        game_data_list.append(block_data)

                    else:
                        shot_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': row[21],
                            'player': row[31] if row[31] != '' else 'null',
                            'type': determine_points(row[37]) if determine_points(row[37]) != '' else 'null',
                            'result': row[35],
                            'season': season,
                            'playoff': playoff
                        }

                        update_vocabs(shot_data)
                        game_data_list.append(shot_data)

                elif row[21] == 'free throw':
                    free_throw_data = {
                        'gameId': game,
                        'roster1': roster1,
                        'roster2': roster2,
                        'time': time,
                        'event': 'shot',
                        'player': row[31] if row[31] != '' else 'null',
                        'type': 'free throw',
                        'result': row[35],
                        'season': season,
                        'playoff': playoff
                    }

                    update_vocabs(free_throw_data)
                    game_data_list.append(free_throw_data)


                elif row[21] == 'rebound':
                    cop = 'cop' if row[37] == 'offensive' else 'nothing'
                    rebound_data = {
                        'gameId': game,
                        'roster1': roster1,
                        'roster2': roster2,
                        'time': time,
                        'event': 'rebound',
                        'player': row[31] if row[31] != '' else 'null',
                        'type': row[37] if row[37] != '' else 'null',
                        'result': cop,
                        'season': season,
                        'playoff': playoff
                    }

                    update_vocabs(rebound_data)
                    game_data_list.append(rebound_data)

                elif row[21] == 'turnover':
                    if row[36] != '':
                        steal_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': 'steal',
                            'player': row[31] if row[31] != '' else 'null',
                            'type': row[36] if row[36] != '' else 'null',
                            'result': 'steal',
                            'season': season,
                            'playoff': playoff
                        }

                        if steal_data['type'] == '':
                            steal_data['type'] = 'null'

                        update_vocabs(steal_data)
                        game_data_list.append(steal_data)

                        turnover_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': 'turnover',
                            'player': row[31] if row[31] != '' else 'null',
                            'type': 'steal',
                            'result': 'cop',
                            'season': season,
                            'playoff': playoff
                        }

                        if turnover_data['type'] == '':
                            turnover_data['type'] = 'null'

                        update_vocabs(turnover_data)
                        game_data_list.append(turnover_data)

                    else:
                        turnover_data = {
                            'gameId': game,
                            'roster1': roster1,
                            'roster2': roster2,
                            'time': time,
                            'event': 'turnover',
                            'player': row[31] if row[31] != '' else 'null',
                            'type': determine_turnover_type(row[36]) if determine_turnover_type(row[36]) != '' else 'null',
                            'result': 'cop',
                            'season': season,
                            'playoff': playoff
                        }

                        update_vocabs(turnover_data)
                        game_data_list.append(turnover_data)

                elif row[21] == 'substitution':
                    sub_data = {
                        'gameId': game,
                        'roster1': roster1,
                        'roster2': roster2,
                        'time': time,
                        'event': 'substitution',
                        'player': row[27] if row[31] != '' else 'null',
                        'type': row[26] if row[26] != '' else 'null',
                        'result': 'substitution',
                        'season': season,
                        'playoff': playoff
                    }

                    update_vocabs(sub_data)
                    game_data_list.append(sub_data)

                elif row[21] == 'foul':

                  player = row[31] if row[31] != '' else 'null'

                  cur_player = player_vocab[player]

                  row_type = ''
                  row_result = ''

                  if cur_player in team1:
                      if final_two(time):
                          team1_bonus = True
                      else:
                          team1_fouls += 1
                          if team1_fouls > 4:
                              team1_bonus = True

                      row_type = determine_foul_type(row[37], team1_bonus)
                      row_result = determine_foul_result(row_type)

                  if cur_player in team2:
                      if final_two(time):
                          team2_bonus = True
                      else:
                          team2_fouls += 1
                          if team2_fouls > 4:
                              team2_bonus = True

                      row_type = determine_foul_type(row[37], team2_bonus)
                      row_result = determine_foul_result(row_type)

                  else:
                      row_result = 'null'

                  foul_data = {
                      'gameId': game,
                      'roster1': roster1,
                      'roster2': roster2,
                      'time': time,
                      'event': 'foul',
                      'player': player,
                      'type': row_type if row_type != '' else 'null',
                      'result': row_result,
                      'season': season,
                      'playoff': playoff
                  }

                  update_vocabs(foul_data)
                  game_data_list.append(foul_data)

                # Save selected games to test folder
        if game in selected_games_indices:
            # Change the extension to .json for the test file
            test_file_name = os.path.splitext(filename)[0] + ".json"
            test_file_path = os.path.join(test_dir, test_file_name)
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)  # Ensure the directory exists

            # Save the game data
            with open(test_file_path, 'w') as test_file:
                json.dump(game_data_list, test_file)
        else:
            # Write non-selected games to the main JSON file
            if not first_entry:
                json_file.write(',\n')
            json.dump(game_data_list, json_file)
            first_entry = False

    json_file.write('\n]\n')  # End of the JSON array

# Save vocabularies
output_dir = "/content/data/languages"
os.makedirs(output_dir, exist_ok=True)

for filename, vocab in vocabularies.items():
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=4)
