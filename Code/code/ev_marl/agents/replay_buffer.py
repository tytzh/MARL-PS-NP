"""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import numpy as np
import random
from collections import namedtuple, deque

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "state_next", "state_total",  "state_total_next",
                                                  "action","reward", "done","i_nbr","w_nbr","i_nbr_next","w_nbr_next","id_mask"])
        self.experience9 = namedtuple("Experience",
                                     field_names=["state", "state_next", "state_total",  "state_total_next",
                                                  "action","reward", "done","id_mask"])
        self.experience_ctcd = namedtuple("Experience",
                                     field_names=["state", "state_next", "action","reward", "done"])
        self.seed = seed
    
    def add(self, state, state_next, state_total, state_total_next, action, reward, done,
            i_nbr,w_nbr,i_nbr_next,w_nbr_next,id_mask):
        """Add a new experience to memory."""
        # state, state_next, state_total, state_total_next, action, action_other, reward, done
        e = self.experience(state, state_next, state_total, state_total_next, action, reward, done,
                            i_nbr,w_nbr,i_nbr_next,w_nbr_next,id_mask)
        temp = e.state_total
        self.memory.append(e)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # including:
        # state, state_next, state_total, state_total_next, action, action_other, reward, done
        experiences = random.sample(self.memory, k=self.batch_size)

        

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states_next = torch.from_numpy(np.vstack([e.state_next for e in experiences if e is not None])).float().to(device)
        states_total = torch.from_numpy(np.vstack([e.state_total for e in experiences if e is not None])).float().to(device)
        states_total_next = torch.from_numpy(np.vstack([e.state_total_next for e in experiences if e is not None])).float().to(
            device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        i_nbr = [e.i_nbr for e in experiences if e is not None]
        i_nbr_next = [e.i_nbr_next for e in experiences if e is not None]
        w_nbr = [e.w_nbr_next for e in experiences if e is not None]
        w_nbr_next = [e.w_nbr_next for e in experiences if e is not None]
        id_mask = [e.id_mask for e in experiences if e is not None]


        return (states, states_next, states_total, states_total_next, actions, rewards, dones, i_nbr, i_nbr_next, w_nbr, w_nbr_next ,id_mask)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
