import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

class CNNActorV3(nn.Module):
    def __init__(self, ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(6, 6), stride=(3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(256 + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, image, vel, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1)
        image = self.conv1(image)
        image = self.conv2(image)
        image = image.view(image.size(0), -1) # Flatten the tensor

        vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)

        combined = torch.cat((image, vel, goal), dim=1)
        combined = self.fc_combined(combined)

        return self.fc_actor(combined)

class CNNCriticV3(nn.Module):
    def __init__(self, ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(6, 6), stride=(3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_action = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(256 + 32 + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, image, vel, goal, action):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1)
        image = self.conv1(image)
        image = self.conv2(image)
        image = image.view(image.size(0), -1) # Flatten the tensor

        vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)
        action = self.fc_action(action)

        combined = torch.cat((image, vel, goal, action), dim=1)
        combined = self.fc_combined(combined)

        return self.fc_actor(combined)
    
