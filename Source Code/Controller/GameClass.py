import pandas as pd

class Game:
    def __init__(self, roster1, roster2, season, playoff):
        self.season = season
        self.playoff = playoff

        # List to hold inputs for the models
        self.inputs = {
            "roster1": [roster1],
            "roster2": [roster2],
            "event": [1],
            "time": [1],
            "player": [1],
            "result": [1],
            "type": [1],
            "season": [self.season],
            "playoff": [self.playoff],
            "current_event": [],
            "current_player": [],
            "current_type": []
        }

        self.load_vocabs()

    def load_vocabs(self):
        # Load vocabularies
        self.rosters_vocab_path = '/content/data/languages/rosters_vocab.json'
        with open(self.rosters_vocab_path, 'r') as f:
            self.rosters_vocab = json.load(f)

    def add_next(self, roster1, roster2, event, time, player, event_type, result):
        self.inputs['roster1'].append(roster1)
        self.inputs['roster2'].append(roster2)
        self.inputs['event'].append(event)
        self.inputs['time'].append(time)
        self.inputs['player'].append(player)
        self.inputs['type'].append(event_type)
        self.inputs['result'].append(result)
        self.inputs['season'].append(self.season)
        self.inputs['playoff'].append(self.playoff)

    def add_current_event(self, event):
        self.inputs['current_event'].append(event)

    def add_current_player(self, player):
        self.inputs['current_player'].append(player)

    def add_current_type(self, event_type):
        self.inputs['current_type'].append(event_type)


    def length(self):
        lengths = {key: len(values) for key, values in self.inputs.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Inconsistent lengths in inputs: {lengths}")
        return next(iter(lengths.values()))

    # Hash function for roster
    def hash_roster(self, roster):
        if len(roster) <= 3:
            return 'start'
        prime = 31
        encoded_value = 0
        for i, num in enumerate(roster):
            encoded_value += num * (prime ** i)
        return self.rosters_vocab[str(encoded_value)]

    def event_prediction(self):
        hashed_roster1 = [self.hash_roster(roster) for roster in self.inputs['roster1']]
        hashed_roster2 = [self.hash_roster(roster) for roster in self.inputs['roster2']]

        input_data = {
            'roster1': hashed_roster1,
            'roster2': hashed_roster2,
            'time': self.inputs['time'],
            'event': self.inputs['event'],
            'player': self.inputs['player'],
            'type': self.inputs['type'],
            'result': self.inputs['result'],
            'season': self.inputs['season'],
            'playoff': self.inputs['playoff']
        }

        return input_data

    def player_prediction(self):
        hashed_roster1 = [self.hash_roster(roster) for roster in self.inputs['roster1']]
        hashed_roster2 = [self.hash_roster(roster) for roster in self.inputs['roster2']]

        input_data = {
            'current_event': self.inputs['current_event'],
            'roster1': hashed_roster1,
            'roster2': hashed_roster2,
            'time': self.inputs['time'],
            'event': self.inputs['event'],
            'player': self.inputs['player'],
            'type': self.inputs['type'],
            'result': self.inputs['result'],
            'season': self.inputs['season'],
            'playoff': self.inputs['playoff']
        }

        return input_data

    def shot_type_prediction(self):
        # Initialize shot_inputs
        shot_inputs = {key: [] for key in self.inputs}

        # Filter the indices for events 1 and 2
        filtered_indices = [
            i for i, event in enumerate(self.inputs["event"])
            if event == 1 or event == 2
        ]

        for key in self.inputs:
            if key == 'current_type' and len(self.inputs[key]) < max(filtered_indices, default=-1) + 1:
                continue
            shot_inputs[key] = [self.inputs[key][i] for i in filtered_indices]

        hashed_roster1 = [self.hash_roster(roster) for roster in shot_inputs['roster1']]
        hashed_roster2 = [self.hash_roster(roster) for roster in shot_inputs['roster2']]

        # Prepare input data for prediction
        input_data = {
            'current_player': shot_inputs['current_player'],
            'roster1': hashed_roster1,
            'roster2': hashed_roster2,
            'time': shot_inputs['time'],
            'type': shot_inputs['type'],
            'player': shot_inputs['player'],
            'result': shot_inputs['result'],
            'season': shot_inputs['season'],
            'playoff': shot_inputs['playoff']
        }

        return input_data

    def shot_result_prediction(self):
        shot_inputs = {key: [] for key in self.inputs}

        # Filter the inputs for events 1 and 2
        filtered_indices = [
            i for i, event in enumerate(self.inputs["event"])
            if event == 1 or event == 2
        ]

        # Populate `shot_inputs` with filtered data
        for key in self.inputs:
            shot_inputs[key] = [self.inputs[key][i] for i in filtered_indices]

        input_data = {
            'current_player': shot_inputs['current_player'],
            'current_type': shot_inputs['current_type'],
            'roster1': shot_inputs['roster1'],
            'roster2': shot_inputs['roster2'],
            'time': shot_inputs['time'],
            'player': shot_inputs['player'],
            'type': shot_inputs['type'],
            'result': shot_inputs['result'],
            'season': shot_inputs['season'],
            'playoff': shot_inputs['playoff']
        }

        return input_data

    def foul_type_prediction(self):
        foul_inputs = {key: [] for key in self.inputs}

        # Filter the inputs for events 1 and 2
        filtered_indices = [
            i for i, event in enumerate(self.inputs["event"])
            if event == 1 or event == 6
        ]

        for key in self.inputs:
            if key == 'current_type' and len(self.inputs[key]) < max(filtered_indices, default=-1) + 1:
                continue
            foul_inputs[key] = [self.inputs[key][i] for i in filtered_indices]

        # Hash rosters
        hashed_roster1 = [self.hash_roster(roster) for roster in foul_inputs['roster1']]
        hashed_roster2 = [self.hash_roster(roster) for roster in foul_inputs['roster2']]

        input_data = {
            'current_player': foul_inputs['current_player'],
            'roster1': hashed_roster1,
            'roster2': hashed_roster2,
            'time': foul_inputs['time'],
            'player': foul_inputs['player'],
            'type': foul_inputs['type'],
            'result': foul_inputs['result'],
            'season': foul_inputs['season'],
            'playoff': foul_inputs['playoff']
        }

        return input_data

    def turnover_type_prediction(self):
        turnover_inputs = {key: [] for key in self.inputs}

        # Filter the inputs for events 1 and 2
        filtered_indices = [
            i for i, event in enumerate(self.inputs["event"])
            if event == 1 or event == 4
        ]

        for key in self.inputs:
            turnover_inputs[key] = [self.inputs[key][i] for i in filtered_indices]

        input_data = {
            'current_player': turnover_inputs['current_player'],
            'roster1': turnover_inputs['roster1'],
            'roster2': turnover_inputs['roster2'],
            'time': turnover_inputs['time'],
            'player': turnover_inputs['player'],
            'type': turnover_inputs['type'],
            'result': turnover_inputs['result'],
            'season': turnover_inputs['season'],
            'playoff': turnover_inputs['playoff']
        }

        return input_data

    def substitution_prediction(self):
        substitute_output = {key: [] for key in self.inputs}

        # Filter the inputs for events 1 and 2
        filtered_indices = [
            i for i, event in enumerate(self.inputs["event"])
            if event == 1 or event == 8
        ]

        for key in self.inputs:
            substitute_output[key] = [self.inputs[key][i] for i in filtered_indices]

        input_data = {
            'current_player': substitute_output['current_player'],
            'roster1': substitute_output['roster1'],
            'roster2': substitute_output['roster2'],
            'time': substitute_output['time'],
            'player': substitute_output['player'],
            'type': substitute_output['type'],
            'result': substitute_output['result'],
            'season': substitute_output['season'],
            'playoff': substitute_output['playoff']
        }

        return input_data

    def to_dataframe(self, for_model="event_time"):
        if for_model == "event_time":
            return pd.DataFrame({
                "roster1": self.inputs["Roster1"],
                "roster2": self.inputs["Roster2"],
                "time": self.inputs["Time"],
                "event": self.inputs["Event"],
                "player": self.inputs["Player"],
                "type": self.inputs["Type"],
                "result": self.inputs["Result"],
                "season": self.inputs["Season"],
                "playoff": self.inputs["Playoff"]
            })
        elif for_model == "player":
            return pd.DataFrame({
                "roster1": self.inputs["Roster1"],
                "roster2": self.inputs["Roster2"],
                "time": self.inputs["Time"],
                "event": self.inputs["Event"],
                "player": self.inputs["Player"],
                "type": self.inputs["Type"],
                "result": self.inputs["Result"],
                "season": self.inputs["Season"],
                "playoff": self.inputs["Playoff"],
                "current_event": self.inputs["CurrentEvent"]
            })
        else:
            raise ValueError(f"Unknown model type: {for_model}")

    def clear_inputs(self):
        for key in self.inputs:
            self.inputs[key] = []
