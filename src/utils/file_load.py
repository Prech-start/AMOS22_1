import json
import os


def load_json(dataset_name, json_name):
    root_path = 'data'
    real_path = os.path.join('..', '..', root_path, dataset_name, json_name)
    return json.load(open(real_path)), os.path.join('..', '..', root_path, dataset_name)
