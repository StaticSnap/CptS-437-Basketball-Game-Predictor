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

class FoulPredictor:
  def __init__(self, epochs=40, data=None, split_index=0):
    self.epochs = epochs
    self.game_length = 200
    self.feature_dims = 8
    self.event_label = []
    self.time_label = []
    self.data=data
    self.split_index = split_index
    self.load_vocabs()

    # file path leading to the trained model.
    file_path = '/content/data/Foul_Predictor.keras'
    if not os.path.isfile(file_path):
      # if the file doesn't exists (aka this is the first time running this function)
      # then begin the preprocessing to get it ready for training.
      print("no model file found, generating new one")
      if data is None:
        print("Data is None!")
    else:
      # otherwise boot up the pre-trained model
      print("File exists. Loading model...")
      self.model = keras.models.load_model(
          file_path, custom_objects={'cast_to_float32': lambda x: tf.cast(x, tf.float32)}
      )

  # open up all of the json voacbularies that this class makes use of as enums (i think)
  def load_vocabs(self):
    self.rosters_vocab_path = '/content/data/languages/rosters_vocab.json'
    with open(self.rosters_vocab_path, 'r') as f:
        self.rosters_vocab = json.load(f)

    self.player_vocab_path = '/content/data/languages/player_vocab.json'
    with open(self.player_vocab_path, 'r') as f:
        self.player_vocab = json.load(f)

    self.event_vocab_path = '/content/data/languages/event_vocab.json'
    with open(self.event_vocab_path, 'r') as f:
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

    # remove the first set of values
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

    #run through each event and remove all events that aren't fouls by using a mask
    for i in range(len(gameIds)):
      length = len(gameIds[i])
      if length > 0:
        mask = (np.array(events[i]) == 1) | (np.array(events[i]) == 6)
        gameIds[i] = np.array(gameIds[i])[mask].tolist()
        rotations1[i] = np.array(rotations1[i])[mask].tolist()
        rotations2[i] = np.array(rotations2[i])[mask].tolist()
        times[i] = np.array(times[i])[mask].tolist()
        times_normalized[i] = np.array(times_normalized[i])[mask].tolist()
        events[i] = np.array(events[i])[mask].tolist()
        players[i] = np.array(players[i])[mask].tolist()
        type_vals[i] = np.array(type_vals[i])[mask].tolist()
        results[i] = np.array(results[i])[mask].tolist()
        seasons[i] = np.array(seasons[i])[mask].tolist()
        playoffs[i] = np.array(playoffs[i])[mask].tolist()

    def map_values(x):
        if x > 2:
            return x - 3
        else:
            return x

    for i in range(len(gameIds)):
      length = len(gameIds[i])
      if length > 0:
        type_vals[i] = [map_values(x) for x in type_vals[i]]

        # Pad inputs
        self.gameId_inputs.append(np.pad(gameIds[i], (0, self.game_length - length), mode='constant'))
        self.roster1_inputs.append(np.pad(rotations1[i], (0, self.game_length - length), mode='constant'))
        self.roster2_inputs.append(np.pad(rotations2[i], (0, self.game_length - length), mode='constant'))
        self.time_inputs.append(np.pad(times_normalized[i], (0, self.game_length - length), mode='constant'))
        self.event_inputs.append(np.pad(events[i], (0, self.game_length - length), mode='constant'))
        self.player_inputs.append(np.pad(players[i], (0, self.game_length - length), mode='constant'))
        self.type_inputs.append(np.pad(type_vals[i], (0, self.game_length - length), mode='constant'))
        self.result_inputs.append(np.pad(results[i], (0, self.game_length - length), mode='constant'))
        self.season_inputs.append(np.pad(seasons[i], (0, self.game_length - length), mode='constant'))
        self.playoff_inputs.append(np.pad(playoffs[i], (0, self.game_length - length), mode='constant'))


    split_index = int(len(self.gameId_inputs) * split_index)
    print("Split_index: ", split_index)

    #split training and testing data

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
    self.train_current_player_inputs = np.array(train_player[1:])

    # Y data
    self.type_label = np.array(train_type[1:])

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
    self.test_current_player_inputs = np.array(test_player[1:])

    # Test labels
    self.type_test_label = np.array(test_type[1:])

    self.fit();

  def fit(self):
    lstm_units = 128
    feature_dim = 52

    # Input Layers
    roster1 = tf.keras.layers.Input(shape=(self.game_length, 1), name='roster1')
    roster2 = tf.keras.layers.Input(shape=(self.game_length, 1), name='roster2')
    time = tf.keras.layers.Input(shape=(self.game_length, 1), name='time')
    player = tf.keras.layers.Input(shape=(self.game_length, 1), name='player')
    current_player = tf.keras.layers.Input(shape=(self.game_length, 1), name='current_player')
    result = tf.keras.layers.Input(shape=(self.game_length, 1), name='result')
    season = tf.keras.layers.Input(shape=(self.game_length, 1), name='season')
    playoff = tf.keras.layers.Input(shape=(self.game_length, 1), name='playoff')

    # Embedding Layers with Masking
    roster1_embedding = tf.keras.layers.Embedding(input_dim=len(self.rosters_vocab)+1, output_dim=16, mask_zero=True)(roster1)
    roster2_embedding = tf.keras.layers.Embedding(input_dim=len(self.rosters_vocab)+1, output_dim=16, mask_zero=True)(roster2)
    player_embedding = tf.keras.layers.Embedding(input_dim=len(self.player_vocab)+1, output_dim=16, mask_zero=True)(player)
    current_player_embedding = tf.keras.layers.Embedding(input_dim=len(self.player_vocab)+1, output_dim=16, mask_zero=True)(current_player)

    # Reshape Embeddings
    roster1_reshape = tf.keras.layers.Reshape((self.game_length, 16))(roster1_embedding)
    roster2_reshape = tf.keras.layers.Reshape((self.game_length, 16))(roster2_embedding)
    player_reshape = tf.keras.layers.Reshape((self.game_length, 16))(player_embedding)
    current_player_reshape = tf.keras.layers.Reshape((self.game_length, 16))(current_player_embedding)

    # Masking for Non-Embedding Inputs
    time_masked = tf.keras.layers.Masking(mask_value=0.0)(time)
    result_masked = tf.keras.layers.Masking(mask_value=0.0)(result)

    # Cast to ensure numeric data types
    def cast_to_float32(x):
        return tf.cast(x, tf.float32)

    time_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(self.game_length, 1))(time_masked)
    result_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(self.game_length, 1))(result_masked)
    season_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(self.game_length, 1))(season)
    playoff_cast = tf.keras.layers.Lambda(cast_to_float32, output_shape=(self.game_length, 1))(playoff)

    all_inputs = tf.keras.layers.Concatenate(axis=-1)([
            roster1_reshape,
            roster2_reshape,
            time_cast,
            player_reshape,
            result_cast,
            season_cast,
            playoff_cast,
            current_player_reshape
        ])

    print("all inputs shape: ", all_inputs.shape)

    # LSTM Layer
    lstm_output = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(all_inputs)

    # Attention Layer with Mask
    plays_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=lstm_units
    )(lstm_output, lstm_output)

    # Outputs
    foul_type_output = tf.keras.layers.Dense(15, activation='softmax', name='foul_output')(plays_attention)

    # Model Definition
    self.model = tf.keras.Model(
        inputs=[roster1, roster2, time, player, result, season, playoff, current_player],
        outputs=[foul_type_output]
    )

    # UNFINISHED
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),
        loss={
            'foul_output': 'sparse_categorical_crossentropy',
        },
        metrics={
            'foul_output': 'accuracy',
        }
     )

    # Callback to track individual losses
    class LossHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.type_loss = []
            self.val_type_loss = []

        def on_epoch_end(self, epoch, logs=None):
            self.type_loss.append(logs['loss'])
            self.val_type_loss.append(logs['val_loss'])

    loss_history = LossHistory()

    self.model.summary()

    # Train the model
    self.model.fit(
        x={
                'time': self.train_time_inputs,
                'roster1': self.train_roster1_inputs,
                'roster2': self.train_roster2_inputs,
                'player': self.train_player_inputs,
                'result': self.train_result_inputs,
                'season': self.train_season_inputs,
                'playoff': self.train_playoff_inputs,
                'current_player': self.train_current_player_inputs
        },
        y={
             'foul_output': self.type_label,
         },
         validation_split=0.2,
         epochs=self.epochs,
         batch_size=16,
         callbacks=[loss_history]
    )
    # Plot individual losses
    plt.figure(figsize=(12, 6))

    # Type Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history.type_loss, label='Training Type Loss')
    plt.plot(loss_history.val_type_loss, label='Validation Type Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Type Output Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    self.model.save('/content/data/Foul_Predictor.keras')

  def predict_foul_type(self, input_data, valid_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
        roster1_input = input_data['roster1']
        roster2_input = input_data['roster2']
        current_player_input = input_data['current_player']
        time_input = input_data['time']
        event_type_input = input_data['type']
        result_input = input_data['result']
        season_input = input_data['season']
        playoff_input = input_data['playoff']

        length = len(roster1_input)

        # Pad inputs to match the game length
        roster1_input = np.pad(roster1_input, (0, self.game_length - length), mode='constant')
        roster2_input = np.pad(roster2_input, (0, self.game_length - length), mode='constant')
        current_player_input = np.pad(current_player_input, (0, self.game_length - length), mode='constant')
        time_input = np.pad(time_input, (0, self.game_length - length), mode='constant')
        event_type_input = np.pad(event_type_input, (0, self.game_length - length), mode='constant')
        result_input = np.pad(result_input, (0, self.game_length - length), mode='constant')
        season_input = np.pad(season_input, (0, self.game_length - length), mode='constant')
        playoff_input = np.pad(playoff_input, (0, self.game_length - length), mode='constant')

        # Create input dictionary for prediction
        input_data_dict = {
            'roster1': np.array([roster1_input]),
            'roster2': np.array([roster2_input]),
            'current_player': np.array([current_player_input]),
            'player': np.array([current_player_input]),
            'time': np.array([time_input]),
            'type': np.array([event_type_input]),
            'result': np.array([result_input]),
            'season': np.array([season_input]),
            'playoff': np.array([playoff_input])
        }

        # Predict using the model
        output = self.model.predict(input_data_dict, batch_size=1, verbose=0)

        # Sample shot type from probabilities
        foul_type_output = self.sample_foul(output[0][length - 1], valid_indices)
        return foul_type_output

  def sample_foul(self, probabilities, valid_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
      probabilities = np.array(probabilities, dtype=np.float64)

      # Mask probabilities to include only valid indices
      mask = np.zeros_like(probabilities, dtype=bool)
      mask[valid_indices] = True
      probabilities = probabilities * mask

      # Normalize probabilities over valid indices
      probabilities = probabilities / np.sum(probabilities)

      # Sample from valid indices
      return np.random.choice(valid_indices, p=probabilities[valid_indices])


  def train(self):
      self.preprocess(self.data, self.split_index)

      correct_type = 0
      total_type = 0

      real_time = []
      predicted_time = []

      for i, game in enumerate(self.test_roster1_inputs):
          # Predict the outputs for the first test input (batch size of 1)
          test_input = {
                  'roster1': self.test_roster1_inputs[i].reshape(1, self.game_length, 1),
                  'roster2': self.test_roster2_inputs[i].reshape(1, self.game_length, 1),
                  'time': self.test_time_inputs[i].reshape(1, self.game_length, 1),
                  'player': self.test_player_inputs[i].reshape(1, self.game_length, 1),
                  'result': self.test_result_inputs[i].reshape(1, self.game_length, 1),
                  'season': self.test_season_inputs[i].reshape(1, self.game_length, 1),
                  'playoff': self.test_playoff_inputs[i].reshape(1, self.game_length, 1),
                  'current_player': self.test_current_player_inputs[i].reshape(1, self.game_length, 1)
          }

          # Perform prediction
          test_output = self.model.predict(test_input, batch_size=1)

          # Extract the type and time outputs
          type_output = test_output[0]

          real_type_label = self.type_test_label[i]

          # Process type output
          for j, prediction in enumerate(type_output):
              if real_type_label[j] != 0:
                  type_prediction = self.sample_foul(prediction)

                  if type_prediction == real_type_label[j]:
                      correct_type += 1

                  total_type += 1

      print("Accuracy:", correct_type/total_type)

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
