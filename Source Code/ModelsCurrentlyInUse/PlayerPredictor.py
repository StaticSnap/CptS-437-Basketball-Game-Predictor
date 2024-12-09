import ijson
import json
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os
import csv
import pickle
import matplotlib.pyplot as plt

tf.keras.config.enable_unsafe_deserialization()

# Hash function for roster
def hash_roster(roster):
    if len(roster) <= 3:
        return 'start'
    prime = 31
    encoded_value = 0
    for i, num in enumerate(roster):
        encoded_value += num * (prime ** i)
    return str(encoded_value)

class PlayerPredictor:
    def __init__(self, epochs=60, data=None, split_index= 0):
        self.epochs = epochs
        self.game_length = 800
        self.feature_dims = 8
        self.event_label = []
        self.time_label = []
        self.load_vocabs()
        self.data = data
        self.split_index = split_index

        file_path = '/content/data/player_predictor.keras'
        if not os.path.isfile(file_path):
            print('hi')
            if data is None:
                print("Data is None!")
        else:
            print("File exists. Loading model...")
            self.model = keras.models.load_model(
                file_path, custom_objects={'cast_to_float32': lambda x: tf.cast(x, tf.float32)}
            )


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

    def preprocess(self, data, split_index):
        # Normalize time
        self.max_time = data['time'].max()
        data['time_normalized'] = data['time'] / self.max_time

        gameId = data['gameId'].values
        roster1 = data['roster1'].values
        roster2 = data['roster2'].values
        time = data['time'].values
        time_normalized = data['time_normalized'].values
        event = data['event'].values
        player = data['player'].values
        type_val = data['type'].values
        result = data['result'].values
        season = data['season'].values
        playoff = data['playoff'].values

        # Split the games at their start token
        gameIds = np.split(gameId.tolist(), (data.index[data['event'] == 1]).tolist())
        rotations1 = np.split(roster1.tolist(), (data.index[data['event'] == 1]).tolist())
        rotations2 = np.split(roster2.tolist(), (data.index[data['event'] == 1]).tolist())
        times = np.split(time.tolist(), (data.index[data['event'] == 1]).tolist())
        times_normalized = np.split(time_normalized.tolist(), (data.index[data['event'] == 1]).tolist())
        events = np.split(event.tolist(), (data.index[data['event'] == 1]).tolist())
        players = np.split(player.tolist(), (data.index[data['event'] == 1]).tolist())
        type_vals = np.split(type_val.tolist(), (data.index[data['event'] == 1]).tolist())
        results = np.split(result.tolist(), (data.index[data['event'] == 1]).tolist())
        seasons = np.split(season.tolist(), (data.index[data['event'] == 1]).tolist())
        playoffs = np.split(playoff.tolist(), (data.index[data['event'] == 1]).tolist())

        self.gameId_inputs = []
        self.roster1_inputs = []
        self.roster2_inputs = []
        self.time_inputs = []
        self.event_inputs = []
        self.player_inputs = []
        self.type_inputs = []
        self.result_inputs = []
        self.season_inputs = []
        self.playoff_inputs = []

        gameIds.pop(0)
        rotations1.pop(0)
        rotations2.pop(0)
        times.pop(0)
        times_normalized.pop(0)
        events.pop(0)
        players.pop(0)
        type_vals.pop(0)
        results.pop(0)
        seasons.pop(0)
        playoffs.pop(0)

        # Pad Games
        for i in range(len(gameIds)):
            length = len(gameIds[i])
            if length > 0:
                # Pad inputs
                self.gameId_inputs.append(np.pad(gameIds[i], (0, self.game_length - length), mode='constant'))
                self.roster1_inputs.append(np.pad(rotations1[i], (0, self.game_length - length), mode='constant'))
                self.roster2_inputs.append(np.pad(rotations2[i], (0, self.game_length - length), mode='constant'))
                self.time_inputs.append(np.pad(times_normalized[i], (0, self.game_length - length), mode='constant'))  # Normalized time
                self.event_inputs.append(np.pad(events[i], (0, self.game_length - length), mode='constant'))
                self.player_inputs.append(np.pad(players[i], (0, self.game_length - length), mode='constant'))
                self.type_inputs.append(np.pad(type_vals[i], (0, self.game_length - length), mode='constant'))
                self.result_inputs.append(np.pad(results[i], (0, self.game_length - length), mode='constant'))
                self.season_inputs.append(np.pad(seasons[i], (0, self.game_length - length), mode='constant'))
                self.playoff_inputs.append(np.pad(playoffs[i], (0, self.game_length - length), mode='constant'))

        split_index = int(len(self.gameId_inputs) * split_index)
        print("Split_index", split_index)

        # Split train and test
        train_gameIds = self.gameId_inputs[:split_index]
        train_roster1 = self.roster1_inputs[:split_index]
        train_roster2 = self.roster2_inputs[:split_index]
        train_time = self.time_inputs[:split_index]
        train_event = self.event_inputs[:split_index]
        train_player = self.player_inputs[:split_index]
        train_type = self.type_inputs[:split_index]
        train_result = self.result_inputs[:split_index]
        train_season = self.season_inputs[:split_index]
        train_playoff = self.playoff_inputs[:split_index]

        test_gameIds = self.gameId_inputs[split_index:]
        test_roster1 = self.roster1_inputs[split_index:]
        test_roster2 = self.roster2_inputs[split_index:]
        test_time = self.time_inputs[split_index:]
        test_event = self.event_inputs[split_index:]
        test_player = self.player_inputs[split_index:]
        test_type = self.type_inputs[split_index:]
        test_result = self.result_inputs[split_index:]
        test_season = self.season_inputs[split_index:]
        test_playoff = self.playoff_inputs[split_index:]

        # Input arrays
        self.train_gameIds_inputs = np.array(train_gameIds[:-1])
        self.train_roster1_inputs = np.array(train_roster1[1:])
        self.train_roster2_inputs = np.array(train_roster2[1:])
        self.train_time_inputs = np.array(train_time[:-1])
        self.train_event_inputs = np.array(train_event[:-1])
        self.train_player_inputs = np.array(train_player[:-1])
        self.train_type_inputs = np.array(train_type[:-1])
        self.train_result_inputs = np.array(train_result[:-1])
        self.train_season_inputs = np.array(train_season[:-1])
        self.train_playoff_inputs = np.array(train_playoff[:-1])
        self.train_current_event_inputs = np.array(train_event[1:])

        # Y data
        self.train_player_label = np.array(train_player[1:])

        # Test arrays
        self.test_gameIds_inputs = np.array(test_gameIds[:-1])
        self.test_roster1_inputs = np.array(test_roster1[1:])
        self.test_roster2_inputs = np.array(test_roster2[1:])
        self.test_time_inputs = np.array(test_time[:-1])
        self.test_event_inputs = np.array(test_event[:-1])
        self.test_player_inputs = np.array(test_player[:-1])
        self.test_type_inputs = np.array(test_type[:-1])
        self.test_result_inputs = np.array(test_result[:-1])
        self.test_season_inputs = np.array(test_season[:-1])
        self.test_playoff_inputs = np.array(test_playoff[:-1])
        self.test_current_event_inputs = np.array(test_event[1:])

        # Test labels
        self.test_player_label = np.array(test_player[1:])

        self.fit()

    def fit(self):
        # Hyperparameters
        lstm_units = 128
        feature_dim = 52

        # Input Layers
        roster1 = tf.keras.layers.Input(shape=(800, 1), name='roster1')
        roster2 = tf.keras.layers.Input(shape=(800, 1), name='roster2')
        time = tf.keras.layers.Input(shape=(800, 1), name='time')
        event = tf.keras.layers.Input(shape=(800, 1), name='event')
        player = tf.keras.layers.Input(shape=(800, 1), name='player')
        event_type = tf.keras.layers.Input(shape=(800, 1), name='type')
        current_event = tf.keras.layers.Input(shape=(800, 1), name='current_event')
        result = tf.keras.layers.Input(shape=(800, 1), name='result')
        season = tf.keras.layers.Input(shape=(800, 1), name='season')
        playoff = tf.keras.layers.Input(shape=(800, 1), name='playoff')

        # Embedding Layers with Masking
        roster1_embedding = tf.keras.layers.Embedding(input_dim=len(self.rosters_vocab)+1, output_dim=16, mask_zero=True)(roster1)
        roster2_embedding = tf.keras.layers.Embedding(input_dim=len(self.rosters_vocab)+1, output_dim=16, mask_zero=True)(roster2)
        player_embedding = tf.keras.layers.Embedding(input_dim=len(self.player_vocab)+1, output_dim=16, mask_zero=True)(player)

        # Reshape Embeddings
        roster1_reshape = tf.keras.layers.Reshape((800, 16))(roster1_embedding)
        roster2_reshape = tf.keras.layers.Reshape((800, 16))(roster2_embedding)
        player_reshape = tf.keras.layers.Reshape((800, 16))(player_embedding)

        # Masking for Non-Embedding Inputs
        time_masked = tf.keras.layers.Masking(mask_value=0.0)(time)
        event_masked = tf.keras.layers.Masking(mask_value=0.0)(event)
        current_event_masked = tf.keras.layers.Masking(mask_value=0.0)(current_event)
        event_type_masked = tf.keras.layers.Masking(mask_value=0.0)(event_type)
        result_masked = tf.keras.layers.Masking(mask_value=0.0)(result)

        # Cast to ensure numeric data types
        def cast_to_float32(x):
            return tf.cast(x, tf.float32)

        time_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(time_masked)
        event_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(event_masked)
        event_type_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(event_type_masked)
        current_event_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(current_event_masked)
        result_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(result_masked)
        season_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(season)
        playoff_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(800, 1))(playoff)

        all_inputs = tf.keras.layers.Concatenate(axis=-1)([
            current_event_cast,
            roster1_reshape,
            roster2_reshape,
            time_cast,
            event_cast,
            player_reshape,
            event_type_cast,
            result_cast,
            season_cast,
            playoff_cast
        ])

        print("All inputs shape:", all_inputs.shape)

        # LSTM Layer
        plays_lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)(all_inputs)

        # Attention Layer with Mask
        plays_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=lstm_units
        )(plays_lstm, plays_lstm)

        # Outputs
        player_output = tf.keras.layers.Dense(len(player_vocab)+1, activation='softmax', name='player_output')(plays_attention)

        # Model Definition
        self.model = tf.keras.Model(
            inputs=[current_event, roster1, roster2, time, event, player, event_type, result, season, playoff],
            outputs=[player_output]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={
                'player_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'player_output': 'accuracy'
            }
        )

        # Callback to track individual losses
        class LossHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.player_loss = []
                self.val_player_loss = []

            def on_epoch_end(self, epoch, logs=None):
                # Fixing the key names to match the logs dictionary
                self.player_loss.append(logs.get('loss', 0))
                self.val_player_loss.append(logs.get('val_loss', 0))


        loss_history = LossHistory()

        self.model.summary()

        # Train the model
        self.model.fit(
            x={
                'time': self.train_time_inputs,
                'event': self.train_event_inputs,
                'roster1': self.train_roster1_inputs,
                'roster2': self.train_roster2_inputs,
                'player': self.train_player_inputs,
                'type': self.train_type_inputs,
                'result': self.train_result_inputs,
                'season': self.train_season_inputs,
                'playoff': self.train_playoff_inputs,
                'current_event': self.train_current_event_inputs
            },
            y={
                'player_output': self.train_player_label
            },
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=16,
            callbacks=[loss_history]
        )

        # Plot individual losses
        plt.figure(figsize=(12, 6))

        # Event Loss
        plt.subplot(1, 2, 1)
        plt.plot(loss_history.player_loss, label='Training Player Loss')
        plt.plot(loss_history.val_player_loss, label='Validation Player Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Player Output Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        self.model.save('/content/data/player_predictor.keras')

    def predict_player(self, input_data, valid_indices=[1, 2, 3, 4, 5, 6, 7, 8]):
        # Extract the data
        roster1_input = input_data['roster1']
        roster2_input = input_data['roster2']
        time_input = input_data['time']
        event_input = input_data['event']
        player_input = input_data['player']
        event_type_input = input_data['type']
        result_input = input_data['result']
        season_input = input_data['season']
        playoff_input = input_data['playoff']
        current_event_input = input_data['current_event']

        length = len(roster1_input)

        # Pad the data
        roster1_input = np.pad(roster1_input, (0, self.game_length - length), mode='constant')
        roster2_input = np.pad(roster2_input, (0, self.game_length - length), mode='constant')
        time_input = np.pad(time_input, (0, self.game_length - length), mode='constant')
        event_input = np.pad(event_input, (0, self.game_length - length), mode='constant')
        player_input = np.pad(player_input, (0, self.game_length - length), mode='constant')
        event_type_input = np.pad(event_type_input, (0, self.game_length - length), mode='constant')
        result_input = np.pad(result_input, (0, self.game_length - length), mode='constant')
        season_input = np.pad(season_input, (0, self.game_length - length), mode='constant')
        playoff_input = np.pad(playoff_input, (0, self.game_length - length), mode='constant')
        current_event_input = np.pad(current_event_input, (0, self.game_length - length), mode='constant')

        roster1_input = np.array([roster1_input])
        roster2_input = np.array([roster2_input])
        time_input = np.array([time_input])
        event_input = np.array([event_input])
        player_input = np.array([player_input])
        event_type_input = np.array([event_type_input])
        result_input = np.array([result_input])
        season_input = np.array([season_input])
        playoff_input = np.array([playoff_input])
        current_event_input = np.array([current_event_input])

        input_data_dict = {
            'current_event': current_event_input,
            'roster1': roster1_input,
            'roster2': roster2_input,
            'time': time_input,
            'event': event_input,
            'player': player_input,
            'type': event_type_input,
            'result': result_input,
            'season': season_input,
            'playoff': playoff_input
        }

        # Use the model to make the actual prediction
        output = self.model.predict(input_data_dict, batch_size=1, verbose=0)

        player_output = self.player_sample(output[0][length-1], valid_indices)

        return player_output

    def player_sample(self, probabilities, valid_indices=None):
        # Convert probabilities to a NumPy array
        probabilities = np.array(probabilities, dtype=np.float64)

        # Default valid indices to all indices except 0 and 1 if not provided
        if valid_indices is None:
            valid_indices = [i for i in range(len(probabilities)) if i not in [0, 1]]

        # Create a mask to zero out probabilities for invalid indices
        mask = np.zeros_like(probabilities, dtype=bool)
        mask[valid_indices] = True

        # Apply the mask to probabilities
        probabilities = probabilities * mask

        # Normalize probabilities for valid indices
        probabilities = probabilities / np.sum(probabilities)

        # Sample from the valid indices
        return np.random.choice(len(probabilities), p=probabilities)

    def train(self):
        self.preprocess(self.data, self.split_index)

        correct_event = 0
        total_event = 0

        real_time = []
        predicted_time = []

        for i, game in enumerate(self.test_roster1_inputs):
            # Predict the outputs for the first test input (batch size of 1)
            test_input = {
                'current_event': self.test_current_event_inputs[i].reshape(1, 800, 1),
                'roster1': self.test_roster1_inputs[i].reshape(1, 800, 1),
                'roster2': self.test_roster2_inputs[i].reshape(1, 800, 1),
                'time': self.test_time_inputs[i].reshape(1, 800, 1),
                'event': self.test_event_inputs[i].reshape(1, 800, 1),
                'player': self.test_player_inputs[i].reshape(1, 800, 1),
                'type': self.test_type_inputs[i].reshape(1, 800, 1),
                'result': self.test_result_inputs[i].reshape(1, 800, 1),
                'season': self.test_season_inputs[i].reshape(1, 800, 1),
                'playoff': self.test_playoff_inputs[i].reshape(1, 800, 1)
            }

            # Perform prediction
            test_output = self.model.predict(test_input, batch_size=1)

            # Extract the player output
            player_output = test_output[0]

            real_player_label = self.test_player_label[i]

            # Process player output
            for j, prediction in enumerate(player_output):
                if self.test_player_label[0][j] != 0:
                    # Define valid indices
                    player_prediction = self.player_sample(prediction)

                    # Use probabilistic sampling
                    if player_prediction == real_player_label[j]:
                        correct_event += 1

                    total_event += 1

        print("Accuracy:", correct_event/total_event)

def read_file(file_path):
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
