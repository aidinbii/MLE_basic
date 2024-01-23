# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = 'settings.json'

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

def load_and_split_data(test_size=0.2, random_state=42):
    # Load the Iris dataset
    df = load_iris(as_frame=True).frame

    # Convert DataFrame to NumPy arrays
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert NumPy arrays back to DataFrames
    df_train = pd.DataFrame(data=X_train, columns=df.columns[:-1])
    df_train['target'] = y_train

    df_test = pd.DataFrame(data=X_test, columns=df.columns[:-1])
    df_test['target'] = y_test

    return df_train, df_test

def save(df: pd.DataFrame, out_path: os.path):
    logger.info(f"Saving data to {out_path}...")
    df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    df_train, df_inference = load_and_split_data()
    save(df_train, TRAIN_PATH)
    save(df_inference, INFERENCE_PATH)
    logger.info("Script completed successfully.")