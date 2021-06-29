"""Deep Q-Networks"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings as s


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s.NUM_STATES, 16)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(16, s.NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN:
    """docstring for DQN"""

    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net = Net()
        self.target_net = Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((s.MEMORY_CAPACITY, s.NUM_STATES * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=s.LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, eps):
        temp = np.zeros(s.NUM_STATES)
        temp[state] = 1
        state = torch.unsqueeze(torch.FloatTensor(temp), 0)

        if np.random.randn() >= eps:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
        else:  # random policy
            action = np.random.randint(0, s.NUM_ACTIONS)

        return action

    def store_transition(self, state, action, reward, next_state):

        temp_1 = np.zeros(s.NUM_STATES)
        temp_1[state] = 1
        state = temp_1

        temp_2 = np.zeros(s.NUM_STATES)
        temp_2[next_state] = 1
        next_state = temp_2

        transition = np.hstack((state, action, reward, next_state))
        index = self.memory_counter % s.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, learning_step, inner_step, ps):

        # update the parameters
        if self.learn_step_counter % s.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(s.MEMORY_CAPACITY, s.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :s.NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, s.NUM_STATES:s.NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, s.NUM_STATES + 1:s.NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -s.NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + s.GAMMA * q_next.max(1)[0].view(s.BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        loss = loss * pow(s.LAMBDA, (learning_step - inner_step)) if ps == 1 else loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
