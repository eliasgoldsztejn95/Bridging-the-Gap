import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

class CNNWithGoal(nn.Module):
    def __init__(self):
        super(CNNWithGoal, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers for goal processing
        self.fc_goal = nn.Linear(2, 64)
        
        # Fully connected layers for combining image and goal information
        self.fc_combined = nn.Linear(32 * 30 * 30 + 64, 128)
        self.fc_actor = nn.Linear(128, 2)  # Output is a 2D vector for velocity
    
    def forward(self, image, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        # Image processing
        image = F.relu(self.conv1(image))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 32 * 30 * 30)

        
        # Goal processing
        goal = F.relu(self.fc_goal(goal))
        
        # Concatenate image and goal information
        combined = torch.cat((image, goal), dim=1)
        
        # Fully connected layers for final prediction
        combined = F.relu(self.fc_combined(combined))
        output = self.fc_actor(combined)
        
        return output

class CNNWithGoalReward(nn.Module):
    def __init__(self):
        super(CNNWithGoalReward, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers for goal and velocity processing
        self.fc_goal = nn.Linear(2, 64)
        self.fc_action = nn.Linear(2, 64)
        
        # Fully connected layers for combining image, goal, and action information
        self.fc_combined = nn.Linear(32 * 30 * 30 + 64 + 64, 128)
        
        # Output layer for reward prediction
        self.fc_critic = nn.Linear(128, 1)
    
    def forward(self, image, goal, action):
        # Image processing
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        image = F.relu(self.conv1(image))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 32 * 30 * 30)
        
        # Goal and velocity processing
        goal = F.relu(self.fc_goal(goal))
        action = F.relu(self.fc_action(action))
        
        # Concatenate image, goal, and velocity information
        combined = torch.cat((image, goal, action), dim=1)
        
        # Fully connected layers for final reward prediction
        combined = F.relu(self.fc_combined(combined))
        reward = self.fc_critic(combined)
        
        return reward

class CNNWithGoalTiny(nn.Module):
    def __init__(self):
        super(CNNWithGoalRewardTiny, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers for goal and velocity processing
        self.fc_combined = nn.Linear(32 * 15 * 15 + 2, 32)
        
        # Output layer for reward prediction
        self.fc_reward = nn.Linear(32, 1)
    
    def forward(self, image, goal):
        # Image processing
        image = F.relu(self.conv1(image))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 32 * 7 * 7)
        
        # Concatenate image, goal, and velocity information directly
        combined = torch.cat((image, goal), dim=1)
        
        # Fully connected layers for final reward prediction
        combined = F.relu(self.fc_combined(combined))
        reward = self.fc_reward(combined)
        
        return reward

class CNNWithGoalRewardTiny(nn.Module):
    def __init__(self):
        super(CNNWithGoalRewardTiny, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers for goal and velocity processing
        self.fc_combined = nn.Linear(32 * 7 * 7 + 2 + 2, 32)
        
        # Output layer for reward prediction
        self.fc_reward = nn.Linear(32, 1)
    
    def forward(self, image, goal, velocity):
        # Image processing
        image = F.relu(self.conv1(image))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 32 * 7 * 7)
        
        # Concatenate image, goal, and velocity information directly
        combined = torch.cat((image, goal, velocity), dim=1)
        
        # Fully connected layers for final reward prediction
        combined = F.relu(self.fc_combined(combined))
        reward = self.fc_reward(combined)
        
        return reward


# class CNNActor(nn.Module):
#     """ Winning algorithm
#     """

#     def __init__(self, ):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6), stride=(3, 3)),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU()
#         )
#         self.fc_goal = nn.Sequential(
#             nn.Linear(2, 32),
#             nn.ReLU()
#         )
#         self.fc_combined = nn.Sequential(
#             nn.Linear(64 + 32, 32),
#             nn.ReLU()
#         )
#         self.fc_actor = nn.Sequential(
#             nn.Linear(32, 2),
#         )

#     def forward(self, image, goal):
#         if image.dim() == 3:
#             image = torch.unsqueeze(image, 1) 
        
#         image = self.conv1(image)
#         image = self.conv2(image)
#         image = self.conv3(image)
#         image = image.view(-1, 64)

#         goal = self.fc_goal(goal)

#         combined = torch.cat((image, goal), dim=1)
#         combined = self.fc_combined(combined)

#         return self.fc_actor(goal)

# class CNNCritic(nn.Module):
#     """ Winnning algorithm
#     """

#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6), stride=(3, 3)),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU()
#         )
#         self.fc_goal = nn.Sequential(
#             nn.Linear(2, 32),
#             nn.ReLU()
#         )
#         self.fc_action = nn.Sequential(
#             nn.Linear(2, 32),
#             nn.ReLU()
#         )
#         self.fc_combined = nn.Sequential(
#             nn.Linear(64 + 32 + 32, 32),
#             nn.ReLU()
#         )
#         self.fc_critic = nn.Sequential(
#             nn.Linear(32, 2),
#         )

#     def forward(self, image, goal, action):
#         if image.dim() == 3:
#             image = torch.unsqueeze(image, 1) 

#         image = self.conv1(image)
#         image = self.conv2(image)
#         image = self.conv3(image)
#         image = image.view(-1, 64)

#         goal = self.fc_goal(goal)
#         action = self.fc_action(action)

#         combined = torch.cat((image, goal, action), dim=1)
#         combined = self.fc_combined(combined)

#         return self.fc_critic(combined)
    

class ActorCNNFine(nn.Module):
    def __init__(self, input_channels=1, goal_dim=2):
        super(ActorCNNFine, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc_combined = nn.Linear(1568 + goal_dim, 64)
        self.fc_actor = nn.Linear(64, 2) 

    def forward(self, image, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1568)
        x = torch.cat((x, goal), dim=1)
        x = F.relu(self.fc_combined(x))
        x = F.relu(self.fc_actor(x))
        return x

class CriticCNNFine(nn.Module):
    def __init__(self, input_channels=1, goal_dim=2, action_dim=2):
        super(CriticCNNFine, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc_combined = nn.Linear(1568 + goal_dim + action_dim, 64)
        self.fc_critic = nn.Linear(64, 2) 

    def forward(self, image, goal, action):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1568)
        x = torch.cat((x, goal, action), dim=1)
        x = F.relu(self.fc_combined(x))
        x = F.relu(self.fc_critic(x))
        return x


class FCActor(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.fc_image = nn.Sequential(
            nn.Linear(60*60, 256),
            nn.ReLU()
        )
        self.fc_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(256 + 32, 32),
            nn.ReLU()
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, image, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 

        image = image.view(-1, 60*60)
        image = self.fc_image(image)

        goal = self.fc_goal(goal)

        combined = torch.cat((image, goal), dim=1)
        combined = self.fc_combined(combined)

        return self.fc_actor(goal)

class FCCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc_image = nn.Sequential(
            nn.Linear(60*60, 256),
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
            nn.Linear(256 + 32 + 32, 32),
            nn.ReLU()
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, image, goal, action):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 

        image = image.view(-1, 60*60)
        image = self.fc_image(image)

        goal = self.fc_goal(goal)
        action = self.fc_action(action)

        combined = torch.cat((image, goal, action), dim=1)
        combined = self.fc_combined(combined)

        return self.fc_actor(combined)


class CNNActor(nn.Module):
    """ Winning algorithm
    """

    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6), stride=(3, 3)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU()
        )
        self.fc_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU()
        )
        self.fc_linear = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_angular = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, image, vel, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        image = self.conv1(image)
        image = self.conv2(image)
        image = self.conv3(image)
        image = image.view(-1, 64)

        #vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)

        combined = torch.cat((image, goal), dim=1)
        combined = self.fc_combined(combined)

        #linear_output = self.fc_linear(combined)
        #angular_output = self.fc_angular(combined)

        #return torch.cat((linear_output, angular_output), dim=1)
        return self.fc_actor(combined)

class CNNCritic(nn.Module):
    """ Winnning algorithm
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6), stride=(3, 3)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 4),
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
            nn.Linear(64 + 32 + 32, 32),
            nn.ReLU()
        )
        self.fc_linear = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_angular = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_critic = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, image, vel, goal, action):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        image = self.conv1(image)
        image = self.conv2(image)
        image = self.conv3(image)
        image = image.view(-1, 64)

        #vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)
        action = self.fc_action(action)

        combined = torch.cat((image, goal, action), dim=1)
        combined = self.fc_combined(combined)

        #linear_output = self.fc_linear(combined)
        #angular_output = self.fc_angular(combined)

        #return torch.cat((linear_output, angular_output), dim=1)
        return self.fc_critic(combined)


##################################################################################################################################
##################################################################################################################################
    
class CNNActorV2(nn.Module):
    """ Winning algorithm
    """

    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU()
        )
        self.fc_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_image = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 32, 32),
            nn.ReLU()
        )
        self.fc_linear = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_angular = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, image, vel, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        image = self.conv1(image)
        image = self.conv2(image)
        image = self.conv3(image)
        image = image.view(-1, 128)

        #vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)

        combined = torch.cat((image, goal), dim=1)
        combined = self.fc_combined(combined)

        #linear_output = self.fc_linear(combined)
        #angular_output = self.fc_angular(combined)

        #return torch.cat((linear_output, angular_output), dim=1)
        return self.fc_actor(combined)

class CNNCriticV2(nn.Module):
    """ Winnning algorithm
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 4),
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
        self.fc_image = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 32 + 32, 32),
            nn.ReLU()
        )
        self.fc_linear = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_angular = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.fc_critic = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, image, vel, goal, action):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1) 
        image = self.conv1(image)
        image = self.conv2(image)
        image = self.conv3(image)
        image = image.view(-1, 128)

        #vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)
        action = self.fc_action(action)

        combined = torch.cat((image, goal, action), dim=1)
        combined = self.fc_combined(combined)

        #linear_output = self.fc_linear(combined)
        #angular_output = self.fc_angular(combined)

        #return torch.cat((linear_output, angular_output), dim=1)
        return self.fc_critic(combined)


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
    
########################
#    NO Dropout
############################

class CNNActorV4(nn.Module):
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
            nn.Linear(128, 64),
            nn.ReLU()
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

class CNNCriticV4(nn.Module):
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
            nn.Linear(128, 64),
            nn.ReLU()
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

#########################
## TWO FRAMES
##########################

class CNNActorV5(nn.Module):
    def __init__(self, ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(6, 6), stride=(3, 3)),
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

class CNNCriticV5(nn.Module):
    def __init__(self, ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(6, 6), stride=(3, 3)),
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

#########################
## THREE CONVS
##########################

class CNNActorV6(nn.Module):
    def __init__(self, ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, 4), stride=(2, 2)),
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU()
        )
        self.fc_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 32, 64),
            nn.ReLU()
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, image, vel, goal):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1)
        #print(f"image 1: {image.shape}")
        image = self.conv1(image)
        #print(f"image 2: {image.shape}")
        image = self.conv2(image)
        #print(f"image 3: {image.shape}")
        image = self.conv3(image)
        #print(f"image 4: {image.shape}")
        image = image.view(image.size(0), -1) # Flatten the tensor
        #print(f"image 5: {image.shape}")
        

        #vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)

        combined = torch.cat((image, goal), dim=1)
        combined = self.fc_combined(combined)

        return self.fc_actor(combined)

class CNNCriticV6(nn.Module):
    def __init__(self, ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, 4), stride=(2, 2)),
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_vel = nn.Sequential(
            nn.Linear(2, 8),
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
            nn.Linear(512 + 32 + 32, 64),
            nn.ReLU()
        )
        self.fc_actor = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, image, vel, goal, action):
        if image.dim() == 3:
            image = torch.unsqueeze(image, 1)
        image = self.conv1(image)
        image = self.conv2(image)
        image = self.conv3(image)
        image = image.view(image.size(0), -1) # Flatten the tensor

        #vel = self.fc_vel(vel)
        goal = self.fc_goal(goal)
        action = self.fc_action(action)

        combined = torch.cat((image, goal, action), dim=1)
        combined = self.fc_combined(combined)

        return self.fc_actor(combined)