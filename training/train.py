"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


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


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
class Training():
    def __init__(self, input_dim) -> None:
        self.model = SimpleClassifier(input_dim=input_dim)

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.test(X_test, y_test)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")

        X = df.iloc[:, :-1]
        y = np.array(df.iloc[:, -1])
        names = df.columns

        # Scale data to have mean 0 and variance 1 
        # which is importance for convergence of the neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data set into training and testing
        return train_test_split(X_scaled, y, test_size=test_size, random_state=conf['general']['random_state'])

    
    def train(self, X_train, y_train) -> None:
        logging.info("Training the model...")
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn   = nn.CrossEntropyLoss()

        # Training loop
        num_epochs = 100

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(X_train_tensor)
            loss = loss_fn(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')


    def test(self, X_test, y_test) -> float:
        logging.info("Testing the model...")
        X_test_tensor = torch.FloatTensor(X_test)

        # Evaluate the model on the test set
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(X_test_tensor)
            _, predicted_classes = torch.max(predictions, 1)
            accuracy = accuracy_score(y_test, predicted_classes.numpy())

        logging.info(f"Accuracy on the test set: {accuracy * 100:.2f}%")
        return accuracy

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pth')
        else:
            path = os.path.join(MODEL_DIR, path)

        # Save the trained model
        torch.save(self.model.state_dict(), path)


def main():
    configure_logging()

    data_proc = DataProcessor()
    df = data_proc.prepare_data()

    tr = Training(df.shape[1] - 1)
    tr.run_training(df, test_size=conf['train']['test_size'])

if __name__ == "__main__":
    main()