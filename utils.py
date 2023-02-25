import os
import pickle
import shutil


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_cluster_in_pickle(filename, encoding_list):
    with open(filename, 'wb') as f:
        pickle.dump(encoding_list, f)


def load_cluster_in_pickle(filename):
    with open(filename, 'rb') as f:
        encoding_list = pickle.load(f)
    return encoding_list


def check_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
