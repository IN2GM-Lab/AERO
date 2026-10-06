import os
import json
import numpy as np
import random
import math

CWND_THRESHOLD = .85

mean_noise = 10.5
std_dev_noise = 4.5
last_cwnd = 0


def scale(value):
    return round(value, 2)

def scale_to_two_digits(number):
    num_digits = len(str(int(number)))
    scale_factor = 10 ** (num_digits - 2)
    scaled_number = number / scale_factor
    return scaled_number
# Function to process a JSON file and append values to metric_values
def get_data(file_path, idx):
    global last_cwnd
    with open(file_path, 'r') as file:
        data = json.load(file)

    all_metrics_values = data.get("observations", [])
    a = np.zeros(15, dtype=int)
    b = np.zeros(15, dtype=int)
    for j in range(15):
        a[j] = j * 10
        b[j] = (j + 1) * 10 - 5
        
    recv_rate = all_metrics_values[idx][a[0]:b[0]]
    queuing_delay = all_metrics_values[idx][a[3]:b[3]]
    delay1 = all_metrics_values[idx][a[4]:b[4]]
    min_seen_delay = all_metrics_values[idx][a[5]:b[5]]
    jitter1 = all_metrics_values[idx][a[9]:b[9]]
    lost_packets1 = all_metrics_values[idx][a[11]:b[11]]
    video1 = all_metrics_values[idx][a[12]:b[12]]
    audio1 = all_metrics_values[idx][a[13]:b[13]]
    probing1 = all_metrics_values[idx][a[14]:b[14]]


    throughput = np.mean(recv_rate)
    if len(str(int(throughput))) > 2:
        throughput = scale_to_two_digits(throughput)
    delay = np.mean(np.abs(delay1)) + random.randint(1,99)
    noise = np.random.normal(mean_noise, std_dev_noise)
    delay = delay + noise
    min_delay = np.mean(min_seen_delay)
    jitter = np.mean(jitter1)
    lost_packets = np.sum(lost_packets1)
    video = np.mean(video1)
    audio = np.mean(audio1)
    probing = np.mean(probing1)

    if lost_packets == 0 :
        lost_packets = random.randint(0,5)


    throughput = scale(throughput)
    cwnd = np.mean(recv_rate) * 60 / 1000
    cwnd = int(cwnd)
    delay = scale(delay)
    min_delay = scale(min_delay)
    jitter = scale(jitter)
    lost_packets = scale(lost_packets)

    true_capacity_values = data.get("true_capacity", [])
    true_capacity = true_capacity_values[idx]
    if math.isnan(true_capacity) :
        true_capacity = last_cwnd
    max_cwnd = CWND_THRESHOLD * (true_capacity * 60 / 1000)
    max_cwnd = int(max_cwnd)
    last_cwnd = max_cwnd

    return throughput, cwnd, max_cwnd, delay, min_delay, jitter, lost_packets, video, audio, probing

'''# Root folder path
root_folder = '/home/ubuntu/Json/'
# Iterate through all folders and files
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(foldername, filename)
            process_json_file(file_path, 0)
            break'''
