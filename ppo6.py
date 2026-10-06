import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        h1_shape = 256
        h2_shape = 256

        self.fc1 = nn.Linear(state_dim, h1_shape)
        self.bn1 = nn.BatchNorm1d(h1_shape)
        self.fc2 = nn.Linear(h1_shape, h2_shape)
        self.bn2 = nn.BatchNorm1d(h2_shape)
        self.fc3 = nn.Linear(h2_shape, action_dim)

        self.stddev = 0.1
        self.explore = 4000
        self.theta = 0.1
        self.step = 0.3

    def forward(self, state, use_noise=False):
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        output = torch.tanh(self.fc3(x))

        if use_noise:
            output_with_noise = torch.zeros_like(output, device=device)
            noise_type = random.randint(1, 6)
            if noise_type == 1:
                ou_noise = self.ornstein_uhlenbeck_noise(output.shape, device)
                output_with_noise = output + ou_noise
            elif noise_type == 2:
                gaussian_noise = torch.normal(mean=torch.zeros_like(output, device=device), std=self.gaussian_decay())
                output_with_noise = output + gaussian_noise
            elif noise_type == 3:
                gaussian_noise = torch.normal(mean=torch.zeros_like(output, device=device), std=self.stddev)
                output_with_noise = output + gaussian_noise
            elif noise_type == 4:
                gaussian_noise = torch.normal(mean=torch.zeros_like(output, device=device), std=self.gaussian_step())
                output_with_noise = output + gaussian_noise
            elif noise_type == 5:
                uniform_noise = torch.rand_like(output, device=device) * 4 - 2
                output_with_noise = output  + uniform_noise

            # Clip the action within the desired range
            output_with_noise = torch.clamp(output_with_noise, -1.0, 1.0)
        
            return output_with_noise.to(device)  # Ensure the tensor is on the same device
        else:
            return output
        
    def ornstein_uhlenbeck_noise(self, shape, device, dt=1):
        dX = self.theta * (torch.zeros(shape, device=device) - torch.zeros(shape, device=device)) + self.stddev * torch.normal(mean=torch.zeros(shape, device=device), std=torch.ones(shape, device=device)) * np.sqrt(dt)
        return dX



    def gaussian_decay(self):
        self.explore *= 0.999  # You may want to adjust this decay rate based on your requirements
        return self.stddev * np.exp(-self.explore / 50000)

    def gaussian_step(self):
        if self.explore > 0:
            self.explore -= self.step
        return self.stddev * np.exp(-self.explore / 50000)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        h1_shape = 256
        h2_shape = 256
        self.fc1 = nn.Linear(state_dim + action_dim, h1_shape)  # Include action_dim in the input
        self.fc2 = nn.Linear(h1_shape, h2_shape)
        self.fc3 = nn.Linear(h2_shape, 1)



    def forward(self, state, action):
       # print(action)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        x = F.leaky_relu(self.fc1(torch.cat([state, action], dim=1)))  # Concatenate state and action
        x = F.leaky_relu(self.fc2(x))
        output = self.fc3(x)
        return output