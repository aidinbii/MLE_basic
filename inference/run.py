"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import time
import sys
from datetime import datetime
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = 'settings.json'

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

class SimpleClassifier(nn.Module):
     def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)
        
     def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str, input_dim: int):
    """Loads and returns the specified model"""
    try:
        logging.info(f'Path of the model: {path}')
        model = SimpleClassifier(input_dim=input_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str):
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        X = df.iloc[:, :-1]

        # Scale data to have mean 0 and variance 1 
        # which is importance for convergence of the neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert data to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled)

        return X, X_tensor

    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, infer_data) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""
    # Perform batch inference
    with torch.no_grad():
        predictions = model(infer_data[1])
        _, predicted_classes = torch.max(predictions, 1)

    infer_data[0]['results'] = predicted_classes.numpy()
    return infer_data[0]


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))

    model = get_model_by_path(get_latest_model_path(), input_dim=infer_data[0].shape[1])
   
    start_time = time.time()
    results = predict_results(model, infer_data)
    end_time = time.time()
    logging.info(f"Inference completed in {end_time - start_time} seconds.")
    store_results(results, args.out_path)
    #logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()