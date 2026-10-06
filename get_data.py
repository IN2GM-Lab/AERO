import os
import json
import numpy as np
from parse_json1 import get_data

root_folder = '/home/ubuntu/Json/'

file_paths = []
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(foldername, filename)
            file_paths.append(file_path)


file_count = 0
idx = 0

while file_count < len(file_paths):
    with open(file_paths[file_count], 'r') as file:
        data = json.load(file)

    all_metrics_values = data.get("observations", [])
    counter = len(all_metrics_values)

    if idx == counter:
        idx = 0
        file_count += 1
        continue

    get_data(file_paths[file_count], idx)
    idx += 1

print("All operations done")