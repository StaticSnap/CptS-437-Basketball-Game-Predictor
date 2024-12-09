import os
from google.colab import drive
import zipfile
import shutil
import json
import os
from tqdm import tqdm

# mount the user's drive so that the data can be read
drive.mount('/content/drive')

# Define the directory path
languages_dir = '/content/data/languages'

# Define the models directory path
models_dir = '/content/data/models'

# Create the directories if they dont exist
os.makedirs(languages_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

test_output_dir = '/content/data/NBAdata-test'
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir, exist_ok=True)

# Path to data
nba_data_zip ='/content/drive/MyDrive/NBAdata/NBAdata-dirty.zip'

# Clear data folder
if os.path.exists('/content/data'):
    for item in os.listdir('/content/data'):
        item_path = os.path.join('/content/data', item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

# Unzip NBAdata-dirty.zip into /content/data/NBAdata-dirty
with zipfile.ZipFile(nba_data_zip, 'r') as zip_ref:
    files = zip_ref.infolist()
    for file in tqdm(files, desc="Extracting files", unit="file"):
        zip_ref.extract(file, '/content/data/NBAdata-dirty')
