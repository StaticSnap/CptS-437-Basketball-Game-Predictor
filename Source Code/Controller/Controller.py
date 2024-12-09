import ijson
import json
from IPython.display import clear_output

class Controller:
    def __init__(self):
        self.data = self.read_file('/content/data/NBADATA.json')
        self.split_index = 0.95
        self.event_predictor = EventPredictor(data=self.data, split_index=self.split_index)
        self.player_predictor = PlayerPredictor(data=self.data, split_index=self.split_index)
        self.shot_type_predictor = ShotTypePredictor(data=self.data, split_index=self.split_index)
        self.shot_result_predictor = ShotResultPredictor(data=self.data, split_index=self.split_index)
        self.foul_type_predictor = FoulPredictor(data=self.data, split_index=self.split_index)
        self.accumulated_results = []

        # self.turnover_type_predictor = TurnoverPredictor(data=self.data, split_index=self.split_index)
        # self.substitution_predictor = SubstitutionPredictor(data=self.data, split_index=self.split_index)
        self.load_vocabs()

    def load_models(self):
        self.event_predictor = EventPredictor(data=self.data, split_index=self.split_index)
        self.player_predictor = PlayerPredictor(data=self.data, split_index=self.split_index)
        self.shot_type_predictor = ShotTypePredictor(data=self.data, split_index=self.split_index)
        self.shot_result_predictor = ShotResultPredictor(data=self.data, split_index=self.split_index)
        self.foul_type_predictor = FoulPredictor(data=self.data, split_index=self.split_index)

    def load_vocabs(self):
        # Load vocabularies
        self.rosters_vocab_path = '/content/data/languages/rosters_vocab.json'
        with open(self.rosters_vocab_path, 'r') as f:
            self.rosters_vocab = json.load(f)

        self.player_vocab_path = '/content/data/languages/player_vocab.json'
        with open(self.player_vocab_path, 'r') as f:
            self.player_vocab = json.load(f)

        self.event_vocab_path = '/content/data/languages/event_vocab.json'
        with open(self.player_vocab_path, 'r') as f:
            self.event_vocab = json.load(f)

        self.type_vocab_path = '/content/data/languages/type_vocab.json'
        with open(self.type_vocab_path, 'r') as f:
            self.type_vocab = json.load(f)

        self.result_vocab_path = '/content/data/languages/result_vocab.json'
        with open(self.result_vocab_path, 'r') as f:
            self.result_vocab = json.load(f)

        self.season_vocab_path = '/content/data/languages/season_vocab.json'
        with open(self.season_vocab_path, 'r') as f:
            self.season_vocab = json.load(f)

    def read_file(self, file_path):
        data = {'gameId': [], 'roster1': [], 'roster2': [], 'time': [], 'event': [], 'player': [], 'type': [], 'result': [], 'season': [], 'playoff': []}

        with open(file_path, 'r') as file:
            items = ijson.items(file, 'item')
            for line in tqdm(items, desc="Processing file", unit="line"):
                for item in line:
                    data['gameId'].append(item['gameId'])
                    data['roster1'].append(rosters_vocab[hash_roster(item['roster1'])])
                    data['roster2'].append(rosters_vocab[hash_roster(item['roster2'])])
                    data['time'].append(item['time'])
                    data['event'].append(event_vocab[item['event']])
                    data['player'].append(player_vocab[item['player']])
                    data['type'].append(type_vocab[item['type']])
                    data['result'].append(result_vocab[item['result']])
                    data['season'].append(season_vocab[item['season']])
                    data['playoff'].append(item['playoff'])
        return pd.DataFrame(data)

    def train_models(self):
        self.event_predictor.train()
        self.player_predictor.train()
        self.shot_type_predictor.train()
        self.shot_result_predictor.train()
        self.foul_type_predictor.train()
        # self.turnover_type_predictor.train()
        # self.substitution_predictor.train()

    def predict_game(self, file_path):
        game_on = True
        season = ''
        playoff = 0
        team1_points = 0
        team2_points = 0
        all_roster1 = set()
        all_roster2 = set()

        with open(file_path, 'r') as file:
            # Load the entire JSON content
            plays = json.load(file)

        # Process the first play
        first_play = plays[0]
        season = first_play['season']
        playoff = first_play['playoff']
        self.roster1_in = first_play['roster1']
        self.roster2_in = first_play['roster2']

        # Process the rest of the plays
        for play in plays:
            all_roster1.update(play['roster1'])
            all_roster2.update(play['roster2'])

            if play['event'] == 'shot':
                points = 0
                if play['result'] == 'made':
                    if play['type'] == '2pt':
                        points = 2
                    elif play['type'] == '3pt':
                        points = 3
                    elif play['type'] == 'free throw':
                        points = 1

                    if play['player'] in self.player_vocab and self.player_vocab[play['player']] in self.roster1_in:
                        team1_points += points
                    elif play['player'] in self.player_vocab and self.player_vocab[play['player']] in self.roster2_in:
                        team2_points += points

        # Determine roster1_out and roster2_out
        self.roster1_out = list(all_roster1 - set(self.roster1_in))
        self.roster2_out = list(all_roster2 - set(self.roster2_in))

        winner_actual = ''
        if team1_points > team2_points:
            winner_actual = 'Team1'
        else:
            winner_actual = 'Team2'

        # Generate Prediction
        self.game = Game(self.roster1_in, self.roster2_in, self.season_vocab[season], playoff)
        end_threshold = 2800
        self.time = 1
        self.team1_pred_score = 0
        self.team2_pred_score = 0

        self.possession = 0

        winner_predicted = ''

        def predict_rebound(self):
            both_teams = self.roster1_in + self.roster2_in

            event_instance = 4

            # Sample player
            player_instance = self.player_predictor.predict_player(self.game.player_prediction(), both_teams)
            self.game.add_current_player(player_instance)

            type_instance = 0
            result_instance = 0
            if player_instance in self.roster1_in:
                if self.possession == 0:
                    type_instance = 20
                    result_instance = 8
                else:
                    type_instance = 21
                    self.possession = 1
                    result_instance = 7
            elif player_instance in self.roster2_in:
                if self.possession == 0:
                    type_instance = 21
                    self.possession = 0
                    result_instance = 7
                else:
                    type_instance = 20
                    result_instance = 8

            self.game.add_current_type(type_instance)
            self.game.add_next(self.roster1_in, self.roster2_in, event_instance, self.time, player_instance, type_instance, result_instance)

        def generate_free_throws(self, point_gain, player):
            for x in range(0, point_gain):
                event_instance = 2
                self.game.add_current_event(event_instance)
                player_instance = player
                self.game.add_current_player(player_instance)
                type_instance = 5
                self.game.add_current_type(type_instance)
                shot_result_instance = self.shot_result_predictor.predict_shot_result(self.game.shot_result_prediction(), [2, 3])

                if shot_result_instance == 2:
                    if self.possession == 0:
                        self.team1_pred_score += 1
                    else:
                        self.team2_pred_score += 1
                shot_result_instance += 1

                self.game.add_next(self.roster1_in, self.roster2_in, event_instance, self.time, player_instance, type_instance, shot_result_instance)

            if self.possession == 0:
                self.possession = 1
            else:
                self.possession = 0

        while game_on:
            event_instance, time_advance = self.event_predictor.predict_event_time(self.game.event_prediction(), [2, 3, 4, 6, 8])
            self.time += time_advance

            # Check for end of game
            if self.time > end_threshold:
                if self.team1_pred_score == self.team2_pred_score:
                    end_threshold += 300
                else:
                    if self.team1_pred_score > self.team2_pred_score:
                        winner_predicted = 'Team1'
                    else:
                        winner_predicted = 'Team2'
                    game_on = False
                    break

            # Event is a shot
            if event_instance == 2:
                self.game.add_current_event(event_instance)

                # Sample player
                if self.possession == 0:
                    player_instance = self.player_predictor.predict_player(self.game.player_prediction(), self.roster1_in)
                    self.game.add_current_player(player_instance)

                else:
                    player_instance = self.player_predictor.predict_player(self.game.player_prediction(), self.roster1_in)
                    self.game.add_current_player(player_instance)


                # Sample shot type
                point_gain = 0
                shot_type_instance = self.shot_type_predictor.predict_shot_type(self.game.shot_type_prediction(), [2, 3])
                if shot_type_instance == 2:
                    point_gain += 2
                else:
                    point_gain += 3
                shot_type_instance += 1

                self.game.add_current_type(shot_type_instance)

                # Sample shot result
                # TODO: Implement blocks
                shot_result_instance = self.shot_result_predictor.predict_shot_result(self.game.shot_result_prediction(), [2, 3])
                if shot_result_instance == 2:
                    if self.possession == 0:
                        self.team1_pred_score += point_gain
                    else:
                        self.team2_pred_score += point_gain
                shot_result_instance += 1

                self.game.add_next(self.roster1_in, self.roster2_in, event_instance, self.time, player_instance, shot_type_instance, shot_result_instance)

                if shot_result_instance == 4:
                    event_instance, time_advance = self.event_predictor.predict_event_time(self.game.event_prediction(), [5, 6])
                    self.time += time_advance
                    self.game.add_current_event(event_instance)

                    # Event is a foul
                    if event_instance == 6:
                        both_teams = self.roster1_in + self.roster2_in

                        # Sample player
                        last_player = player_instance
                        player_instance = self.player_predictor.predict_player(self.game.player_prediction(), both_teams)
                        self.game.add_current_player(player_instance)

                        type_instance = 0
                        # Sample foul
                        if player_instance in self.roster1_in:
                            if self.possession == 0:
                                type_instance = self.foul_type_predictor.predict_foul_type(self.game.foul_type_prediction(), [6, 10])
                            else:
                                type_instance = self.foul_type_predictor.predict_foul_type(self.game.foul_type_prediction(), [4, 6, 10])
                        else:
                            if self.possession == 0:
                                type_instance = self.foul_type_predictor.predict_foul_type(self.game.foul_type_prediction(), [4, 6, 10])
                            else:
                                type_instance = self.foul_type_predictor.predict_foul_type(self.game.foul_type_prediction(), [6, 10])

                        type_instance += 3
                        self.game.add_current_type(type_instance)

                        if (type_instance == 7):
                            result_instance = 10
                            self.game.add_next(self.roster1_in, self.roster2_in, event_instance, self.time, player_instance, type_instance, result_instance)
                            generate_free_throws(self, point_gain, last_player)
                        else:
                            result_instance = 13
                            self.game.add_next(self.roster1_in, self.roster2_in, event_instance, self.time, player_instance, type_instance, result_instance)
                            if player_instance in self.roster1_in:
                                if self.possession == 0:
                                    self.possession = 1
                                else:
                                    self.possession = 0
                    # Event is a rebound
                    else:
                        predict_rebound(self)
                else:
                    if self.possession == 0:
                        self.possession = 1
                    else:
                        self.possession = 0

            elif event_instance == 3:
                self.time += time_advance
                pass
            elif event_instance == 4:
                self.time += time_advance
                pass
            elif event_instance == 6:
                self.time += time_advance
                pass
            elif event_instance == 8:
                self.time += time_advance
                pass

        # After game ends, add to accumulated results
        result = f"File: {file_path}, Winner Actual: {winner_actual}, Winner Predicted: {winner_predicted}"
        self.accumulated_results.append(result)

        return {
            'team1_points_actual': team1_points,
            'team2_points_actual': team2_points,
            'team1_points_predicted': self.team1_pred_score,
            'team2_points_predicted': self.team2_pred_score,
            'winner_actual': winner_actual,
            'winner_predicted': winner_predicted
        }


    def test(self, file_path='/content/data/NBAdata-test'):
        team1_actual_scores = []
        team2_actual_scores = []
        team1_predicted_scores = []
        team2_predicted_scores = []
        winner_actual_list = []
        winner_predicted_list = []

        # Iterate through each file in the directory
        for filename in os.listdir(file_path):
            file_full_path = os.path.join(file_path, filename)

            # Ensure it's a file and not a subdirectory
            if os.path.isfile(file_full_path):
                print(f"Processing file: {file_full_path}")
                result = self.predict_game(file_full_path)

                print(result)

                # Collect scores for standard deviation calculation
                team1_actual_scores.append(result['team1_points_actual'])
                team2_actual_scores.append(result['team2_points_actual'])
                team1_predicted_scores.append(result['team1_points_predicted'])
                team2_predicted_scores.append(result['team2_points_predicted'])

                # Collect actual and predicted winners
                winner_actual_list.append(result['winner_actual'])
                winner_predicted_list.append(result['winner_predicted'])

        # Calculate standard deviations
        team1_std_dev = np.std(np.array(team1_actual_scores) - np.array(team1_predicted_scores))
        team2_std_dev = np.std(np.array(team2_actual_scores) - np.array(team2_predicted_scores))

        # Calculate percentage of correct predictions
        correct_predictions = sum(1 for actual, predicted in zip(winner_actual_list, winner_predicted_list) if actual == predicted)
        total_predictions = len(winner_predicted_list)
        correct_percentage = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        # Print results
        print("\nSummary:")
        print(f"Standard Deviation (Team 1 Scores): {team1_std_dev:.2f}")
        print(f"Standard Deviation (Team 2 Scores): {team2_std_dev:.2f}")
        print(f"Percentage of Correct Predictions: {correct_percentage:.2f}%")

controller = Controller()
controller.train_models()
controller.test()
