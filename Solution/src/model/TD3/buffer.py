import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mem_size = max_size
        self.batch_size = batch_size
        self.memory_init()

    def memory_init(self):
        self.mem_cnt = 0
        self.state_memory = np.zeros((self.mem_size, self.state_dim))
        self.action_memory = np.zeros((self.mem_size, self.action_dim))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, self.state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self, flag=None):
        mem_len = min(self.mem_cnt, self.mem_size)
        batch_size = min(mem_len, self.batch_size)
        if flag == 'all':
            batch = np.random.choice(mem_len, mem_len, replace=False)
        else:
            batch = np.random.choice(mem_len, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt >= self.batch_size
