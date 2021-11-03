############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#hard = , 1606600271, 1606601629, 1606600335, 1606601684
# hardest = 1606598990
#       medium:    1606415443,
#       easy:      1606420347 , 29, 1606599639,  , 1606600422
#win = 1606420347, 1606599639, 1606604170, 1606604495, 1606604225, 1606436201,
# 1606604762,1606437508, 1606415217, 1606606352,1606608679, 1606608443, 1606608390
#lose = 1606605686, 1606605914, 1606605937, 1606605914
# idk =  1606604225, 1606604762, 1606606500, 1606608359
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import collections
import random
import math


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 200
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # FOLLOWING ARE MINE
        # set dqn
        self.dqn = DQN()
        # set replay buffer
        self.replay_buffer = ReplayBuffer()
        # set mini-batch size
        self.minibatch_size = 500
        # set target network update every N steps
        self.target_network_update = 20
        # set exploration parameter
        self.epsilon = 1
        # set episode count
        self.episode_number = 0
        self.start_time = None
        self.training = True
        # greedy check after 8 mins
        # loss
        self.loss = []
        self.greedy_path =[]
        self.network = None
        # greedy policy flag
        self.greedy_episode = False
        self.greedy_episode_steps = 0
        self.nn =None

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.num_steps_taken % self.episode_length == 0:
            # turn on greedy episode flag
            if self.episode_number>20 and self.episode_number %3 ==0 and self.training==True:
                self.greedy_episode =True
                self.episode_length = 100
                self.network = self.dqn.q_network.state_dict()
                self.nn = Network(input_dimension=2, output_dimension=4)
            else: #if not greedy episode turn off greedy flag and step count
                self.greedy_episode_steps = 0
                self.greedy_episode=False
            self.greedy_path =[]
            self.episode_number +=1
            if self.episode_number <50:
                self.epsilon = 1
            else:
                self.epsilon =0.3
            #self.episode_length *= .85
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        self.state = state
        if self.training:
            # greedy policy for episodes number > 10, every 3rd episode and
            if self.greedy_episode:
                self.greedy_episode_steps +=1
                discrete_action = self.get_greedy_next_action(self.state)
            elif self.episode_number<30:
                discrete_action = self._choose_epsilon_greedy_next_action(epsilon=1)
            elif 30<= self.episode_number < 50:
                discrete_action = self._choose_epsilon_greedy_next_action(epsilon=self.epsilon)
                self.epsilon *= 0.999
            elif 50<= self.episode_number <= 100:
                discrete_action = self._choose_epsilon_greedy_next_action(epsilon=self.epsilon)
                if self.epsilon<0.7:
                    self.epsilon *= 1.01
            else:
                discrete_action = self._choose_epsilon_greedy_next_action(epsilon=0.23)

        else:
            discrete_action = self._choose_next_action()
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        #self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = discrete_action

        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if self.greedy_episode == True and distance_to_goal <0.01 and self.greedy_episode_steps <100:
            self.network = self.dqn.q_network.state_dict()
            self.nn = Network(input_dimension=2, output_dimension=4)
            print("EARLY SROPPING")
            print("111111111111111111111111111111")
            print('**************************')
            print('network weight')
            #print(self.dqn.q_network.state_dict())
            self.greedy_episode = False
            print('greedy steps')
            print(time.time() - self.start_time)
            print(self.greedy_episode_steps)
            self.training = False
        # Convert the distance to a reward
        reward = (((math.sqrt(2) - distance_to_goal ))**2)*.5
        if not np.any(self.state - next_state):
            reward /= 1.5
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # draw path
        if self.greedy_episode:
            self.greedy_path.append(self.state)
        # network update
        if self.training and self.greedy_episode ==False:
            self.replay_buffer.add_transition(transition)
            if len(self.replay_buffer.buffer) > self.minibatch_size:                    #training buffer start parameter
                transitions = self.replay_buffer.sample_mini_batch(self.minibatch_size)
                loss = self.dqn.train_q_network(transitions)
                if self.num_steps_taken % self.episode_length == 0:
                    self.loss.append(loss)
            if self.num_steps_taken % self.target_network_update ==0:
                self.dqn.update_target_weights()
        self.state = next_state

    def _choose_next_action(self):
        return random.randint(0,3)

    def get_greedy_action(self, state):
        self.network = self.dqn.q_network.state_dict()
        self.nn = Network(input_dimension=2, output_dimension=4)
        state = torch.tensor(state)
        self.nn.load_state_dict(self.network)
        q_values_array = self.nn(state).detach().numpy()
        best_action = np.argmax(q_values_array)
        continuous_action = self._discrete_action_to_continuous(best_action)
        return continuous_action

    def get_discrete_greedy_action(self, state):
        state = torch.tensor(state)
        self.nn.load_state_dict(self.network)
        q_values_array = self.nn(state).detach().numpy()
        best_action = np.argmax(q_values_array)
        return best_action

    def get_greedy_episode_next_action(self, state):
        state = torch.tensor(state)
        self.nn.load_state_dict(self.network)
        q_values_array = self.nn(state).detach().numpy()
        best_action = np.argmax(q_values_array)
        return best_action

    def get_greedy_next_action(self, state):
        state = torch.tensor(state)
        q_values_array = self.dqn.q_network(state).detach().numpy()
        best_action = np.argmax(q_values_array)
        return best_action

    def _choose_epsilon_greedy_next_action(self, epsilon):  # greedy policy
        best_action = self.get_greedy_next_action(self.state)
        if np.random.uniform(0, 1) > epsilon:
            return best_action
        else:
            other_actions = [0,1,2,3]
            action = random.choices(other_actions, weights=[2,1,0,1],k=1)[0]
            return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        # clockwise
        if discrete_action == 0:
            # Move right: 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 1:
            # Move up: 0 to the right, and 0.1 downwards
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        if discrete_action == 2:
            # Move left: 0.1 to the left, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 3:
            # Move down: 0 to the right, and 0.1 upwards
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        return continuous_action



# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a target Q-network
        self.target_q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.

        loss = self._calculate_loss(transition)
        #print('loss is ' + str(loss))
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition, gamma=0.99):
        states, actions, rewards, next_states = zip(*transition)
        state_tensor = torch.tensor(states, dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        # this is my network's prediction for Belmman - gets q value for specific action taken in given state
        network_prediction = self.q_network.forward(state_tensor).gather(dim=1,
                                                                         index=action_tensor.unsqueeze(-1)).squeeze(-1)
        # q values for next state
        # q_next_state = self.target_q_network.forward(next_states_tensor)
        # get action with max value for next state across the mini batch
        max_a_next_state = torch.amax(self.target_q_network.forward(next_states_tensor).detach(), 1)
        # network's label - network's prediction
        loss = torch.nn.MSELoss()(reward_tensor + 0.9 * max_a_next_state, network_prediction)
        return loss

    def update_target_weights(self):
        network_weights = self.q_network.state_dict()
        self.target_q_network.load_state_dict(network_weights)

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=5000)

    def add_transition(self, transition):
        self.buffer.append(transition)

    def sample_mini_batch(self, n):
        minibatch_idx = np.random.choice(len(self.buffer), n, replace=False)
        # mini-batch is list of 4-tuples
        mini_batch =[self.buffer[i] for i in minibatch_idx]
        return mini_batch