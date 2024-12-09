from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import ijson

class ShotResultNeuralNetwork:
    def __init__(self, data, split_index):
        self.data = data
        self.split_index = split_index

    def preprocess(self):
        # Normalize 'time'
        scaler = StandardScaler()
        time_column = scaler.fit_transform(self.data[['time']]).astype(np.float32)

        # Separate target column before preprocessing
        y = self.data['result']

        # One-hot encode categorical features
        categorical_features = ['type', 'season', 'player']
        categorical_data = pd.get_dummies(self.data[categorical_features], drop_first=True)

        # Combine normalized numerical features with encoded categorical features
        X = categorical_data

        # One-hot encode the target column
        y = pd.get_dummies(y).to_numpy()

        # Split into training and testing sets
        self.X_train, self.X_test = X.iloc[:self.split_index].to_numpy(), X.iloc[self.split_index:].to_numpy()
        self.y_train, self.y_test = y[:self.split_index], y[self.split_index:]

        # Split self.y_test into two halves
        half_index = len(self.y_test) // 2
        self.X_test_data = self.y_test[half_index:]
        self.y_test_data = self.y_test[half_index:]
        self.X_test = self.X_test[half_index:]
        self.y_test = self.y_test[:half_index]

        # # PCA for 'team' and 'defense'
        # defense_features_scaled = scaler.fit_transform(pd.DataFrame(self.data['defense'].tolist()))
        # team_features_scaled = scaler.fit_transform(pd.DataFrame(self.data['team'].tolist()))

        # pca = PCA(n_components=1)
        # self.data['defense'] = pca.fit_transform(defense_features_scaled)
        # self.data['team'] = pca.fit_transform(team_features_scaled)

        print(self.data.dtypes)

    def fit(self, epochs=50, batch_size=32):
        # Define the neural network
        self.model = Sequential([
            Dense(64, input_dim=self.X_train.shape[1], activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Print the model summary
        self.model.summary()


        # Fit the model
        history = self.model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_test, self.y_test),
                            epochs=epochs,
                            batch_size=batch_size)

        # Save the model for future use
        self.model.save("Shot_Predictor_Neural_Network.keras")

        return history, self.model

    def predict(self, model_path="Shot_Predictor_Neural_Network.keras"):
        print(self.model.predict(self.X_test_data))

def read_file(file_path):
    player_vocab_path = '/content/data/languages/player_vocab.json'
    with open(player_vocab_path, 'r') as f:
        player_vocab = json.load(f)

    # add team and defense as columns
    data = {'time': [], 'event': [], 'player': [], 'type': [], 'result': [], 'season': [], 'playoff': []}

    with open(file_path, 'r') as file:
        items = ijson.items(file, 'item')
        for line in tqdm(items, desc="Processing file", unit="line"):
            for item in line:
                # NOT MESSING WITH ROSTERS

                # # Remove player from team
                # team = item['roster1'][:]
                # if player_vocab[item['player']] in team:
                #     team.remove(player_vocab[item['player']])
                #     data['defense'].append(item['roster2'])
                # else:
                #     team = item['roster2'][:]
                #     if player_vocab[item['player']] in team:
                #         team.remove(player_vocab[item['player']])
                #         data['defense'].append(item['roster1'])
                #     else:
                #         print(item['player'])
                #         print("roster1:", item['roster1'])
                #         print("roster2:", item['roster2'])
                #         print("player:", item['player'])
                #         print("player encoded:", player_vocab[item['player']])

                # data['team'].append(team)

                data['time'].append(item['time'])
                data['event'].append(item['event'])
                data['player'].append(item['player'])
                data['type'].append(item['type'])
                data['result'].append(item['result'])
                data['season'].append(item['season'])
                data['playoff'].append(item['playoff'])

    df = pd.DataFrame(data)
    df = df[df['event'] == 'shot']
    df = df[df['player'] != 'null']
    df = df.drop(columns=['event'])

    # Print unique entries in 'type' and 'result'
    print("Unique entries in 'type':", df['type'].unique())
    print("Unique entries in 'result':", df['result'].unique())

    return df

# Read the data
data = read_file('/content/data/NBADATA.json')

# Initialize the class
split_index = int(len(data) * 0.8)  # 80% for training, 20% for testing
nn = ShotResultNeuralNetwork(data, split_index)

# Preprocess the data
nn.preprocess()

# Fit the neural network
history, model = nn.fit(epochs=50, batch_size=32)
