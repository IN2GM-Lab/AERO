import logging

import socket
import struct
import time
from collections import deque
import numpy as np
import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import random
from aioquic.quic.congestion.ppo5  import Actor, Critic
from typing import Iterable

from ..packet_builder import QuicSentPacket
from .base import (
    K_MINIMUM_WINDOW,
    QuicCongestionControl,
    QuicRttMonitor,
    register_congestion_control,
)


K_LOSS_REDUCTION_FACTOR = 0.5


latest_rtt = 0
smoothed_rtt = 0 
lost_count = 0
congestion_window = 0 
bytes_in_flight = 0


RANDOM_SEED = 42
S_DIM = 4
A_DIM = 1
LEARNING_RATE_ACTOR = 0.0003
LEARNING_RATE_CRITIC = 0.0003
UPDATE_INTERVAL = 500
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6
MIN_MOMERY = 100
LOSS_PENALTY = 5
DELAY_PENALTY = 3
JITTER_PENALTY = 0.1
MAX_CWND = 4640000
MIN_CWND = 40000
TEST_TRACES_VALID = './cooked_test_traces/'
SUMMARY_DIR = './Results/sim/ppo'
LOG_FILE = './Results/sim/ppo/log'
TEST_LOG_FOLDER = './Results/sim/ppo/test_results/'

LOG_FILE_VALID = './Results/sim/ppo/test_results/log_valid_ppo'
TEST_LOG_FOLDER_VALID = './Results/sim/ppo/test_results/'
Log_path = './Results/sim/ppo'

EXTRA_REWARD = 0.10
EXTRA_PENALTY = 0.10
CWND_DIFF_THRESHOLD = 5000

MAX = 1
MIN = 0
LAST_LOST_COUNT = 0
MAX_THR = 0
MAX_DELAY = 0
MAX_LOST = 0
MAX_JITTER = 0
MAX_SELU = 0
MAX_DELTA = 0
MAX = 0
MIN_DELAY = 1000
MIN_THR = 1000
MIN_JITTER = 1000
MIN_LOST = 1000
MIN_SELU = 1000
MIN_DELTA = 10000000
MIN = 0
MAX = 0

updated_cwnd = 1000
exp_cwnd = 1000
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


s_batch = [np.zeros((S_DIM))]

model = Actor(S_DIM,A_DIM).type(dtype)
test_model='/aioquic/src/aioquic/quic/congestion/abr_ppo_38000.model'
model.eval()
reward = 0
model.load_state_dict(torch.load(test_model, map_location=torch.device('cpu')))
min_delay = 0
LAST_LOST_COUNT = 0
congestion_windows = []
# Accept a connection from a client

def scale_to_three_digits(number):
    num_digits = len(str(number))
    scale_factor = 10 ** (num_digits - 3)
    scaled_number = number / scale_factor
    return scaled_number

def scale_to_one_digit(number):
    num_digits = len(str(number))
    scale_factor = 10 ** (num_digits - 1)
    scaled_number = number / scale_factor
    return scaled_number


def get_congestion_window(latest_rtt, smoothed_rtt, lost_count, congestion_window, bytes_in_flight):
    reward = 0
    global LAST_LOST_COUNT
    global MAX, MIN, LAST_LOST_COUNT, MAX_THR, MAX_DELAY, MAX_LOST, MAX_JITTER, MAX_SELU, MAX_DELTA, MIN_DELAY, MIN_THR, MIN_JITTER, MIN_LOST, MIN_SELU, MIN_DELTA
    if len(s_batch) == 0:
            state = [np.zeros((S_DIM))]
    else:
        state = np.array(s_batch[-1], copy=True)
    latest_rtt = latest_rtt * 1000
    smoothed_rtt = smoothed_rtt * 1000
    # dequeue history record
   # state = np.roll(state, -1)

    if latest_rtt!=0:
       throughput = int(bytes_in_flight / latest_rtt)
    else :
       throughput = int(bytes_in_flight)
    if throughput > 999:
        throughput = scale_to_three_digits(throughput)
    delay = smoothed_rtt
    throughput = throughput 
    

    delay=delay/10
    print("Delay:",delay)


    jitter = abs(latest_rtt - smoothed_rtt)
    jitter = scale_to_one_digit(jitter)

    lost_count = lost_count - LAST_LOST_COUNT
    LAST_LOST_COUNT = LAST_LOST_COUNT + lost_count

    lost_packets = lost_count



    if throughput > MAX_THR:
     MAX_THR = throughput

    if delay > MAX_DELAY:
     MAX_DELAY = delay
                    
    if lost_packets > MAX_LOST:
     MAX_LOST = lost_packets
                    
    if jitter > MAX_JITTER :
     MAX_JITTER = jitter  

    if throughput < MIN_THR:
        MIN_THR = throughput

    if delay < MIN_DELAY:
        MIN_DELAY = delay
                    
    if lost_packets < MIN_LOST:
        MIN_LOST = lost_packets
                    
    if jitter < MIN_JITTER:
        MIN_JITTER = jitter 
    
    
    MAX_DELTA = MAX_DELTA / 1000
    MIN_DELTA = MIN_DELTA / 1000

    MAX= np.max([MAX_THR, MAX_JITTER, MAX_DELAY, MAX_LOST])
    MIN = np.min([MIN_THR, MIN_DELAY, MIN_JITTER, MIN_LOST])


    throughput = ((throughput-MIN) / (MAX-MIN))
    delay = ((delay-MIN) / (MAX-MIN)) 
    jitter = ((jitter-MIN) / (MAX-MIN)) 
    lost_packets = ((lost_packets-MIN) / (MAX-MIN)) 

    state[0] = throughput 
    state[1] = delay
    state[2] = jitter
    state[3] = lost_packets 
   # state[4] = delta_cwnd
    
    reward = reward + ((throughput - LOSS_PENALTY * lost_packets - DELAY_PENALTY * delay))

    '''if (reward > 0):
       exp_cwnd = congestion_window * 2
    else:
       exp_cwnd = congestion_window / 2'''
    

    print("Throughput:", throughput)
    print("Delay:", delay)
    print("Lost Packets:", lost_packets)
    print("Congestion Window:",congestion_window)
    print("Reward",reward)

    state = torch.from_numpy(state)

    with torch.no_grad():
         alpha= model(state.unsqueeze(0).type(dtype)).squeeze()
    
    alpha = float(alpha.item())
    if reward > 0 and alpha < 0 :
       alpha = alpha * -1
    if reward < 0 and alpha > 0 :
       alpha = alpha * -1
    print("Tensor:",alpha)

    print("Alpha: ",alpha)

    #updated_cwnd = (1.2 ** alpha) * congestion_window
    if congestion_window != 0 :
       #updated_cwnd = (math.exp(alpha)) * congestion_window
       updated_cwnd = (1.87 ** alpha) * congestion_window
    else:
       #updated_cwnd = (math.exp(alpha))
       updated_cwnd = (1.87** alpha) * congestion_window 
    if congestion_window < 100 :
       updated_cwnd = (1.87 ** alpha) * congestion_window * 3
    updated_cwnd = int(updated_cwnd)
    if updated_cwnd > MAX_CWND :
         updated_cwnd = MAX_CWND
    
    if updated_cwnd < MIN_CWND:
       updated_cwnd = MIN_CWND

    return updated_cwnd



class AeroCongestionControl(QuicCongestionControl):
    """
    Aero congestion control.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("AERO")
    def __init__(self, *, max_datagram_size: int) -> None:
        super().__init__(max_datagram_size=max_datagram_size)
        self._max_datagram_size = max_datagram_size
        self._congestion_recovery_start_time = 0.0
        self._congestion_stash = 0
        self._rtt_monitor = QuicRttMonitor()


    def on_packet_acked(self, *, now: float, packet: QuicSentPacket) -> None:
        global bytes_in_flight
        global congestion_window
        global latest_rtt
        global smoothed_rtt
        global lost_count
        self.bytes_in_flight -= packet.sent_bytes
        bytes_in_flight = self.bytes_in_flight
        logging.info("Bytes in flight: %s" % bytes_in_flight)
        # don't increase window in congestion recovery
        if packet.sent_time <= self._congestion_recovery_start_time:
            return

        if self.ssthresh is None or self.congestion_window < self.ssthresh:
            # slow start
            #self.congestion_window += packet.sent_bytes
            self.congestion_window = get_congestion_window(latest_rtt, smoothed_rtt, lost_count, congestion_window, bytes_in_flight)
            congestion_window = self.congestion_window
            logging.info("Congestion Window: %s" % congestion_window)
        else:
            # congestion avoidance
            self._congestion_stash += packet.sent_bytes
            count = self._congestion_stash // self.congestion_window
            if count:
                self._congestion_stash -= count * self.congestion_window
                self.congestion_window += count * self._max_datagram_size
                congestion_window = self.congestion_window
                logging.info("Congestion Window: %s" % congestion_window)

    def on_packet_sent(self, *, packet: QuicSentPacket) -> None:
        global bytes_in_flight
        self.bytes_in_flight += packet.sent_bytes
        bytes_in_flight = self.bytes_in_flight
        logging.info("Bytes in flight: %s" % bytes_in_flight)

    def on_packets_expired(self, *, packets: Iterable[QuicSentPacket]) -> None:
        global bytes_in_flight
        for packet in packets:
            self.bytes_in_flight -= packet.sent_bytes
        bytes_in_flight = self.bytes_in_flight
        logging.info("Bytes in flight: %s" % bytes_in_flight)

    def on_packets_lost(self, *, now: float, packets: Iterable[QuicSentPacket]) -> None:
        lost_largest_time = 0.0
        global bytes_in_flight
        global congestion_window
        global latest_rtt
        global smoothed_rtt
        global lost_count
        for packet in packets:
            self.bytes_in_flight -= packet.sent_bytes
            lost_largest_time = packet.sent_time
        bytes_in_flight = self.bytes_in_flight
        logging.info("Bytes in flight: %s" % bytes_in_flight)
        # start a new congestion event if packet was sent after the
        # start of the previous congestion recovery period.
        if lost_largest_time > self._congestion_recovery_start_time:
            self._congestion_recovery_start_time = now
            self.congestion_window = get_congestion_window(latest_rtt, smoothed_rtt, lost_count, congestion_window, bytes_in_flight)
            '''self.congestion_window = max(
                int(self.congestion_window * K_LOSS_REDUCTION_FACTOR),
                K_MINIMUM_WINDOW * self._max_datagram_size,
            )'''
            congestion_window = self.congestion_window
            logging.info("Congestion Window: %s" % congestion_window)
            self.ssthresh = self.congestion_window

        # TODO : collapse congestion window if persistent congestion

    def on_rtt_measurement(self, *, now: float, rtt: float, srtt: float) -> None:
        # check whether we should exit slow start
        global latest_rtt
        global smoothed_rtt
        if self.ssthresh is None and self._rtt_monitor.is_rtt_increasing(
            now=now, rtt=rtt
        ):
            self.ssthresh = self.congestion_window
    
    def get_loss(self, *, lost_packets: int) -> None:
        global lost_count
        lost_count = lost_packets
        logging.info("Lost Count: %s" % lost_count)

    def get_rtt(self, *, rtt: float, srtt: float) -> None:
        global latest_rtt
        global smoothed_rtt
        latest_rtt = rtt
        smoothed_rtt = srtt
        logging.info("Latest RTT: %s" % latest_rtt)
        logging.info("Smoothed RTT: %s" % smoothed_rtt)


register_congestion_control("aero", AeroCongestionControl)
