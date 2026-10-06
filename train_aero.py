import argparse
import os
import numpy as np
import random
import json
from tqdm import tqdm
import copy

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from ppo6 import Actor, Critic
from parse_json import get_data

RANDOM_SEED = 42
S_DIM = 4
A_DIM = 1
LEARNING_RATE_ACTOR = 0.0003
LEARNING_RATE_CRITIC = 0.003
UPDATE_INTERVAL = 500
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6
MIN_MEMORY = 100
LOSS_PENALTY = 1
DELAY_PENALTY = 3
JITTER_PENALTY = 2
EXTRA_REWARD = 0.10
EXTRA_PENALTY = 0.10
CWND_DIFF_THRESHOLD = 500

MAX_THR = 0
MAX_DELAY = 0
MAX_LOST = 0
MAX_JITTER = 0
MAX_DELTA = 0
MIN_DELAY = 0
MIN_THR = 0
MIN_JITTER = 0
MIN_LOST = 0
MIN_DELTA = 0
MIN = 0
MAX = 0
BETA = 2
ALPHA_FACTOR = 1.2
LOWER_LIMIT = 1.2
UPPER_LIMIT = 3
new_min = -1
new_max = 1
MAX_SELU = 1
MIN_SELU = -1
LOG_FILE = './Results/sim/ppo4/log'
root_folder = '/home/ubuntu/Json/'
delay1 = 0
LOG_FILE_VALID = './Results/sim/ppo4/test_results/log_valid_ppo'

def scale_to_one_digits(number):
    num_digits = len(str(int(number)))
    scale_factor = 10 ** (num_digits - 1)
    scaled_number = number / scale_factor
    return int(scaled_number)

def scale_to_two_digits(number):
    num_digits = len(str(int(number)))
    scale_factor = 10 ** (num_digits - 2)
    scaled_number = number / scale_factor
    return int(scaled_number)


parser = argparse.ArgumentParser(description='priority_ppo')
parser.add_argument('--test', action='store_true', help='Evaluate only')

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory) > self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            print("Lenght:",len(self.memory))
            return None

        samples = random.sample(self.memory, batch_size)

        # Reshape each state tensor before concatenating
        batch_states = torch.cat([state.view(1, -1) for state, *_ in samples], dim=0)
        batch_actions = torch.cat([action.view(1, -1) for _, action, *_ in samples], dim=0)
        batch_returns = torch.cat([returns.view(1, -1) for _, _, returns, *_ in samples], dim=0)
        batch_advantages = torch.cat([advantages.view(1, -1) for _, _, _, advantages in samples], dim=0)

        return batch_states, batch_actions, batch_returns, batch_advantages


def train():
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    with open('drl_cc.txt', 'r') as file:
        all_lines = file.readlines()
    with open(LOG_FILE + '_record', 'w') as log_file, open(LOG_FILE + '_test', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)

        model_actor = Actor(S_DIM, A_DIM).type(dtype)
        model_critic = Critic(S_DIM, A_DIM).type(dtype)

        model_actor.train()
        model_critic.train()

        optimizer_actor = optim.RMSprop(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.RMSprop(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)

        criterion_critic = torch.nn.SmoothL1Loss()

        state = np.zeros(S_DIM)
        state = torch.from_numpy(state)


        epoch = 0
        time_stamp = 0

        exploration_size = 29
        episode_steps = 25
        update_num = 6
        batch_size = 128
        gamma = 0.95
        gae_param = 0.90
        clip = 0.2
        ent_coeff = 2.6
        memory = ReplayMemory(exploration_size * episode_steps)
        idx = 0
        file_count = 0
        DELAY_PENALTY = 3
        JITTER_PENALTY = 2
        MAX_THR = 0
        MAX_DELAY = 0
        MAX_LOST = 0
        MAX_JITTER = 0
        MAX_DELTA = 0
        MAX = 0
        MIN_DELAY = 1000
        MIN_THR = 1000
        MIN_JITTER = 1000
        MIN_LOST = 1000
        MIN_DELTA = 1000
        MIN = 0
        ALPHA_FACTOR = 1.2
        scaling_factor = 1
        reward = 0
        new_min = -1
        new_max = 1
        MAX_SELU = 1
        MIN_SELU = -1

        while True:
            for explore in range(exploration_size):
                states = []
                rewards = []
                values = []
                returns = []
                actions = []
                advantages = []

                for step in range(episode_steps):
                    
                    with torch.no_grad():
                        model_actor.eval()
                        alpha = model_actor(state.unsqueeze(0).type(dtype)).squeeze()
                        model_actor.train()

                    '''if alpha > MAX_SELU:
                        MAX_SELU = alpha
                    if alpha < MIN_SELU:
                        MIN_SELU = alpha
                    if alpha > 0:
                        sign = +1
                    else :
                        sign = -1
                    if alpha > 1:
                        num_digits = len(str(int(alpha)))
                        alpha /= 10 ** (num_digits)
                    if alpha < -1:
                        alpha = (alpha-MIN_SELU) * sign / (MAX_SELU - MIN_SELU)
                    #print("Alpha:", alpha)
                    if alpha == 0:
                        alpha = sign * random.uniform(0,0.5)
                        alpha = torch.tensor(alpha, dtype=torch.float32)'''
                    reward = 0
                    
                    action_tensor = alpha.unsqueeze(0).unsqueeze(1).to(device=device)
                    v = model_critic(state.unsqueeze(0).type(dtype), action_tensor).detach().cpu()
                    values.append(v)
                    
                    #print("State:",state)
                    states.append(copy.deepcopy(state))
                    #print("States",states)
                    actions.append(torch.tensor(alpha, dtype=torch.float32))

                    # Assuming line contains the values
                    line = all_lines[idx % len(all_lines)]
                    curr_values = line.split()

                    throughput, expected_cwnd, max_cwnd, delay, min_delay, jitter, \
                    lost_packets, video, audio, probing = map(float, curr_values)

                    idx += 1

                    orig_throughput = throughput

                    
                    delay = scale_to_one_digits(int(delay))
                    pred_cwnd = int((ALPHA_FACTOR ** alpha.item()) * expected_cwnd)
    
                    if(pred_cwnd > max_cwnd) :
                        reward = reward - 2*EXTRA_PENALTY
                        ALPHA_FACTOR = ALPHA_FACTOR - 0.1
                                                                
                    delta_cwnd = np.abs(pred_cwnd - expected_cwnd)

                    if expected_cwnd == pred_cwnd :
                        reward = reward + 1.5*EXTRA_REWARD
                    elif expected_cwnd < pred_cwnd :
                        if delta_cwnd < CWND_DIFF_THRESHOLD:
                            reward = reward - EXTRA_PENALTY
                            ALPHA_FACTOR = ALPHA_FACTOR - 0.05
                        else:
                            reward = reward - 2*EXTRA_PENALTY
                            ALPHA_FACTOR = ALPHA_FACTOR - 0.1
                    else:
                        if delta_cwnd < CWND_DIFF_THRESHOLD:
                            reward = reward + EXTRA_REWARD
                            ALPHA_FACTOR = ALPHA_FACTOR + 0.05
                        else:
                            reward = reward + 2*EXTRA_REWARD
                            ALPHA_FACTOR = ALPHA_FACTOR + 0.1

                    if ALPHA_FACTOR > UPPER_LIMIT :
                        ALPHA_FACTOR = UPPER_LIMIT
                    if ALPHA_FACTOR < LOWER_LIMIT :
                        ALPHA_FACTOR = LOWER_LIMIT


                    if video > audio and video > probing:
                        LOSS_PENALTY = 5
                        DELAY_PENALTY = 4
                    elif audio > probing:
                        LOSS_PENALTY = 4
                        DELAY_PENALTY = 3
                    else:
                        LOSS_PENALTY = 2
                        DELAY_PENALTY = 3

                    delta_cwnd = scale_to_two_digits(delta_cwnd)

                    if throughput > MAX_THR :
                        MAX_THR = throughput
                    if delay > MAX_DELAY:
                        MAX_DELAY = delay
                    
                    if lost_packets > MAX_LOST:
                        MAX_LOST = lost_packets
                    
                    if jitter > MAX_JITTER:
                        MAX_JITTER = jitter  
                    
                    if delta_cwnd > MAX_DELTA:
                        MAX_DELTA = delta_cwnd

                    if throughput < MIN_THR:
                        MIN_THR = throughput

                    if delay < MIN_DELAY:
                        MIN_DELAY = delay
                    
                    if lost_packets < MIN_LOST:
                        MIN_LOST = lost_packets
                    
                    if jitter < MIN_JITTER:
                        MIN_JITTER = jitter 

                    if delta_cwnd < MIN_DELTA:
                        MIN_DELTA = delta_cwnd

                    if MAX_DELTA > MAX_DELAY and MAX_DELTA > MAX_JITTER and MAX_DELTA > MAX_LOST and MAX_DELTA > MAX_THR:
                        MAX = MAX_DELTA
                    elif MAX_DELAY > MAX_JITTER and MAX_DELAY > MAX_LOST and MAX_DELAY > MAX_THR:
                        MAX = MAX_DELAY
                    elif MAX_JITTER > MAX_LOST and MAX_JITTER > MAX_THR:
                        MAX = MAX_JITTER
                    elif MAX_LOST > MAX_THR :
                        MAX = MAX_LOST
                    else:
                        MAX = MAX_THR

                    

                   

                    '''if min_delay <= delay and delay <= BETA * min_delay:
                        delay1 = min_delay
                    else:
                        delay1 = delay'''

                    if MIN_DELTA < MIN_DELAY and MIN_DELTA < MIN_JITTER and MIN_DELTA < MIN_LOST and MIN_DELTA < MIN_THR:
                        MIN = MIN_DELTA
                    elif MIN_DELAY < MIN_JITTER and MIN_DELAY < MIN_LOST and MIN_DELAY < MIN_LOST:
                        MIN = MIN_DELAY
                    elif MIN_JITTER < MIN_LOST and MIN_JITTER < MIN_THR:
                        MIN = MIN_JITTER
                    elif MIN_LOST < MIN_THR:
                        MIN = MIN_LOST
                    else:
                        MIN = MIN_THR
                    
 
                    orig_delta_cwnd = delta_cwnd
                    orig_delay = delay
                    orig_loss = lost_packets
                    orig_jitter = jitter
                    
                    throughput = (throughput-MIN) / (MAX-MIN)
                    delta_cwnd = (delta_cwnd-MIN) / (MAX-MIN)
                    delay = (delay-MIN) / (MAX-MIN)
                    jitter = (jitter-MIN) / (MAX-MIN)
                    lost_packets = (lost_packets-MIN) / (MAX-MIN)
                    reward = reward + ((throughput - LOSS_PENALTY * lost_packets - DELAY_PENALTY * delay - JITTER_PENALTY * jitter))

                    rewards.append(reward)

                    #state = np.roll(state, -1)
                    state[0] = throughput
                    state[1] = delay
                    # state[2, -1] = min_delay
                    state[2] = jitter
                    state[3] = lost_packets
                    # state[5, -1] = video
                    # state[6, -1] = audio
                    # state[7, -1] = probing
                
                    #print("State:",state)
                    #state = torch.from_numpy(state)

                    if log_file.tell() == 0:
                        log_file.write(
                            "Alpha_Factor               \tPredicted CWND               \tExpected CWND               \tDeltaCwnd               \tThroughput               \tDelay                    \tMinDelay                    \tJitter                   \tLost Packets             \tReward                   \tAlpha                    \n"
                        )

                    # Assuming a fixed width of 25 characters for each column
                    log_file.write(
                        f"{ALPHA_FACTOR:<25}\t{pred_cwnd:<25}\t{expected_cwnd:<25}\t{delta_cwnd:<25}\t{throughput:<25}\t{delay:<25}\t{min_delay:<25}\t{jitter:<25}\t{lost_packets:<25}\t{reward:<25}\t{alpha:<25}\n"
                    )
                    log_file.flush()

                R = torch.zeros(1, 1)
                v = model_critic(state.unsqueeze(0).type(dtype), action_tensor).detach().cpu()
                R = v.data

                values.append(Variable(R))

                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(rewards))):
                    td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                    A = float(td) + gamma * gae_param * A
                    advantages.insert(0, A)
                    R = A + values[i]
                    returns.insert(0, R)

                memory.push([states, actions, returns, advantages])
                #print(states)
            model_actor_old = Actor(S_DIM, A_DIM).type(dtype)
            model_critic_old = Critic(S_DIM, A_DIM).type(dtype)
            model_actor_old.load_state_dict(model_actor.state_dict())
            model_critic_old.load_state_dict(model_critic.state_dict())
            
            for update_step in range(update_num):
                model_actor.zero_grad()
                model_critic.zero_grad()
                #print("Hello")
                #print(model_actor.parameters())

                # new mini_batch
                # priority_batch_size = int(memory.get_capacity()/10)
                batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)
                # batch_size = memory.return_size()
                # batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)
                
                # old_prob
                probs_old = model_actor_old(batch_states.type(dtype))
                v_pre_old = model_critic_old(batch_states.type(dtype), batch_actions.type(dtype))
                prob_value_old = probs_old
                
                # new prob
                probs = model_actor(batch_states.type(dtype))
                
                #print("Probs old:",probs)
                #print("Probs:", probs)
                v_pre = model_critic(batch_states.type(dtype), batch_actions.type(dtype))
                prob_value = probs
                print(probs)
                # ratio
                ratio = prob_value / (1e-6 + prob_value_old)

                ## non-clip loss
                # surrogate_loss = ratio * batch_advantages.type(dtype)


                # clip loss
                #clipped_advantages = torch.clamp(batch_advantages, -clip, clip)
                surr1 = ratio * batch_advantages.type(dtype)  # surrogate from conservative policy iteration
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages.type(dtype)
                loss_clip_actor = -torch.mean(torch.min(surr1, surr2))
                # value loss
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
                # entropy
                
                probs_tensor = torch.tensor(probs, dtype=torch.float32, requires_grad=False).type(dtype).clone().detach()
                #print("Probs Tensor:",probs_tensor)
                epsilon = 1e-5
                loss_ent = ent_coeff * torch.mean(probs_tensor * torch.log(torch.clamp(probs_tensor, epsilon, 1.0)))
                # total
                print("Lossent:",loss_ent)
                policy_total_loss = loss_clip_actor + loss_ent
                # print("hell0")

                # copy the new model to old model?
                # model_actor_old.load_state_dict(model_actor.state_dict())
                # model_critic_old.load_state_dict(model_critic.state_dict())

                # update 
                print("Policy loss:", policy_total_loss)
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                policy_total_loss.backward()
                # loss_clip_actor.backward(retain_graph=True)
                loss_value.backward()
                optimizer_actor.step()
                
                optimizer_critic.step()
                # print("hell0")

            epoch += 1
            print("Epoch: ", epoch)
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(policy_total_loss.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(loss_ent.detach().cpu().numpy()))

            if epoch % UPDATE_INTERVAL == 0:
                logging.info("Model saved in file")
                add_str = 'ppo'
                model_save_path = LOG_FILE_VALID + "/%s_%s_%d.model" % (str('abr'), add_str, int(epoch))
                torch.save(model_actor.state_dict(), model_save_path)
                critic_save_path = LOG_FILE_VALID + "/%s_%s_%d.model" % (str('abr_critic'), add_str, int(epoch))
                torch.save(model_critic.state_dict(), critic_save_path)
                # entropy_weight = 0.95 * entropy_weight
                ent_coeff = 0.95 * ent_coeff


def main():

    train()


if __name__ == '__main__':
    main()
