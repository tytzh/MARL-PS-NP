""""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

import numpy as np
import random
import copy
from training_utils import extract_by_indices_tensor
from actor_critic import Actor, Critic, ActorC, CriticC
from replay_buffer import ReplayBuffer
from ou_noise import OrnsteinUhlenbeckNoise

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 64      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3             # for soft update of target parameters


LR_ACTOR_B = 1e-4      # learning rate of the actor
LR_CRITIC_B = 1e-4    # learning rate of the critic
WEIGHT_DECAY = 1e-4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # L2 weight decay
LR_ACTOR_C = 1e-5      # learning rate of the actor
LR_CRITIC_C = 1e-5      # learning rate of the critic
WEIGHT_DECAY_C = 1e-4


# setting for other method
# BUFFER_SIZE = int(5e5)  # replay buffer size
# BATCH_SIZE = 64      # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3             # for soft update of target parameters
#
#
# LR_ACTOR_B = 1e-2       # learning rate of the actor
# LR_CRITIC_B = 1e-2     # learning rate of the critic
# WEIGHT_DECAY = 1e-4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # L2 weight decay
# LR_ACTOR_C = 1e-3        # learning rate of the actor
# LR_CRITIC_C = 1e-3      # learning rate of the critic
# WEIGHT_DECAY_C = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_b():
    """Interacts with and learns from the environment."""

    def __init__(self, id_agent, state_size, state_total_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.state_total_size = state_total_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.id_agent = id_agent
        self.seed = random_seed

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR_B ,weight_decay=WEIGHT_DECAY)
        self.actor_masks = []

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_total_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_total_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC_B ,weight_decay=WEIGHT_DECAY)  
        self.critic_masks = []  

        # Noise process for each agent
        self.noise = OrnsteinUhlenbeckNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def initial_state_dict(self):
        
        self.dict_actor = self.actor_target.state_dict()
        self.dict_critic = self.critic_target.state_dict()

    def get_mask_actor(self, model, percent, list_masked):
        weight = model.fc1.weight.data
        total_params = weight.numel()
        all_indices = torch.arange(total_params)

        num_to_freeze = int(percent * 0.01 * total_params)
        if num_to_freeze == 0:
            mask = [[], []]
            return mask, list_masked

        available_indices = torch.tensor(
            list(set(all_indices.numpy()) - set(list_masked.numpy()))
        )

        
        available_weights = weight.flatten()[available_indices]
        sorted_order = torch.argsort(available_weights.abs(), descending=True)
        selected = sorted_order[:num_to_freeze]
        random_indices = available_indices[selected]

        rows, cols = np.unravel_index(random_indices.numpy(), weight.shape)
        mask = [list(rows), list(cols)]
        list_masked = torch.cat((list_masked, random_indices), dim=0)
        return mask, list_masked

    def get_mask_critic(self, model, percent, list_masked):
        weight = model.fcs1.weight.data
        total_params = weight.numel()
        all_indices = torch.arange(total_params)

        num_to_freeze = int(percent * 0.01 * total_params)
        if num_to_freeze == 0:
            mask = [[], []]
            return mask, list_masked

        available_indices = torch.tensor(
            list(set(all_indices.numpy()) - set(list_masked.numpy()))
        )

        
        available_weights = weight.flatten()[available_indices]
        sorted_order = torch.argsort(available_weights.abs(), descending=True)
        selected = sorted_order[:num_to_freeze]
        random_indices = available_indices[selected]

        rows, cols = np.unravel_index(random_indices.numpy(), weight.shape)
        mask = [list(rows), list(cols)]
        list_masked = torch.cat((list_masked, random_indices), dim=0)
        return mask, list_masked

    def make_prune_masks2(self, n_mask, percent,list_actor_masked,list_critic_masked):
        
        model_critic = self.critic_target
        model_actor = self.actor_target

        for i in range(n_mask):
            
            mask_actor_prune,list_actor_masked= self.get_mask_actor(self.actor_target, percent, list_actor_masked)
            mask_critic_prune,list_critic_masked = self.get_mask_critic(self.critic_target, percent, list_critic_masked)

            self.actor_masks.append(mask_actor_prune)
            self.critic_masks.append(mask_critic_prune)

        
        actor_freeze = [ [] for _ in range(n_mask) ]
        critic_freeze = [[] for _ in range(n_mask)]
        for i in range(n_mask):
            listi = list(range(0,n_mask))
            # --> actor
            freezi = [[],[]]
            if i in listi:
                listi.remove(i)
            for j in listi:
                actor_i = self.actor_masks[j]
                freezi[0].extend(actor_i[0])
                freezi[1].extend(actor_i[1])
            actor_freeze[i] = freezi

            # --> critic
            freezi = [[],[]]
            for j in listi:
                critic_i = self.critic_masks[j]
                freezi[0].extend(critic_i[0])
                freezi[1].extend(critic_i[1])
            critic_freeze[i] = freezi
        self.actor_freeze = actor_freeze
        self.critic_freeze = critic_freeze


    def step_mask3(self, state, state_next, state_total, state_total_next, action, reward, done,
            id_mask,weight_ls):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # state, state_next, state_total, state_total_next, action, action_other, reward, done
        self.memory.add(state, state_next, state_total, state_total_next, action, reward, done,
                        1, 1, 1,weight_ls, id_mask)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            # print("Learning")
            experiences = self.memory.sample()
            self.learn_mask3(experiences)

    def act_mask2(self, state, idx_mask, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        actor_freeze = self.actor_freeze[idx_mask]
        row_freeze = actor_freeze[0]
        col_freeze = actor_freeze[1]

        # ================= DEBUG =================
        debug_prune = False
        if random.random() < 0.001:
            debug_prune = True

        # if debug_prune:
        #     print("\n========== ACT MASK DEBUG ==========")
        #     print("idx_mask:", idx_mask)
        #     print("num frozen weights:", len(row_freeze))
        #     print("state shape:", state.shape)
        # =========================================

        original_values = self.actor_local.fc1.weight[row_freeze, col_freeze].clone()

        acts = np.zeros((self.num_agents, self.action_size))

        self.actor_local.eval()

        with torch.no_grad():
            
            if debug_prune:
                if self.num_agents > 1:
                    action_no_mask = self.actor_local(state[0, :]).cpu().data.numpy()
                else:
                    action_no_mask = self.actor_local(state).cpu().data.numpy()

            
            self.actor_local.fc1.weight[row_freeze, col_freeze] = 0

            if self.num_agents > 1:
                for agent in range(self.num_agents):
                    acts[agent, :] = self.actor_local(state[agent, :]).cpu().data.numpy()
            else:
                acts[0, :] = self.actor_local(state).cpu().data.numpy()

            # if debug_prune:
            #     action_masked = acts.copy()
            #     print("action_no_mask:", action_no_mask)
            #     print("action_masked:", action_masked)
            #     print("action_diff:", np.abs(action_masked - action_no_mask).mean())
            #     print("====================================\n")

        self.actor_local.train()

        if add_noise:
            acts += self.noise.sample()

        with torch.no_grad():
            self.actor_local.fc1.weight[row_freeze, col_freeze] = original_values

        return np.clip(acts, -1, 1)


    def reset(self):
        self.noise.reset()

    def extract_data_train2(self, states, states_next, states_total, states_total_next, actions, rewards, dones, idx_mask, mask_weight):

        def assign_to_class(probabilities,idx_mask):
            if probabilities[idx_mask] == max(probabilities):  
                return idx_mask
            else:
                return np.random.choice(len(probabilities), p=probabilities)  

        assigned_classes = [assign_to_class(probs,idx_mask) for probs in mask_weight]
        indices = [index for index, value in enumerate(assigned_classes) if value == idx_mask]

        
        states0 = extract_by_indices_tensor(states, indices)
        states_next0 = extract_by_indices_tensor(states_next, indices)
        states_total0 = extract_by_indices_tensor(states_total, indices)
        states_total_next0 = extract_by_indices_tensor(states_total_next, indices)
        actions0 = extract_by_indices_tensor(actions, indices)
        rewards0 = extract_by_indices_tensor(rewards, indices)
        dones0 = extract_by_indices_tensor(dones, indices)

        return states0, states_next0, states_total0, states_total_next0, actions0, rewards0, dones0

    def learn_mask3(self, experiences):
        
        original_actor_local = copy.deepcopy(self.actor_local.state_dict())
        original_critic_local = copy.deepcopy(self.critic_local.state_dict())

        
        states, states_next, states_total, states_total_next, actions, rewards, dones, _, _, _, mask_weight, id_mark = experiences
        row_freeze_all_actor, col_freeze_all_actor, row_freeze_all_critic, col_freeze_all_critic = self.get_freeze_sharing()

        
        actions_next = self.actor_target(states_total_next[:, 0:self.state_size])
        Q_targets_next = self.critic_target(states_total_next, actions_next)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states_total, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        with torch.no_grad():
            grad_critic_local1 =  self.critic_local.fcs1.weight.grad.clone()  
            grad_critic_local1[row_freeze_all_critic, col_freeze_all_critic ] = 0
            self.critic_local.fcs1.weight.grad = grad_critic_local1
        
        grads_critic_clone = {}
        for name, param in self.critic_local.named_parameters():
            if param.grad is not None:  
                grads_critic_clone[name] = param.grad.clone()

        
        actions_pred = self.actor_local(states_total[:, 0:self.state_size])
        actor_loss = -self.critic_local(states_total, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        with torch.no_grad():
            grad_actor_local1 =  self.actor_local.fc1.weight.grad.clone()  
            grad_actor_local1[row_freeze_all_actor, col_freeze_all_actor ] = 0
            self.actor_local.fc1.weight.grad = grad_actor_local1
        
        grads_actor_clone = {}
        for name, param in self.actor_local.named_parameters():
            if param.grad is not None:  
                grads_actor_clone[name] = param.grad.clone()


        
        n_group = len(mask_weight[0])
        for mask_i in range(n_group):
            grad_actor_local_i,grad_critic_local_i = self.update_special_part_withmask(original_actor_local, original_critic_local, id_mark, states,
                                         states_next,states_total, states_total_next, actions, rewards, dones, mask_i,mask_weight)
            
            grads_actor_clone['fc1.weight'] = grads_actor_clone['fc1.weight'] + grad_actor_local_i
            grads_critic_clone['fcs1.weight'] = grads_critic_clone[
                                                    'fcs1.weight'] + grad_critic_local_i

        
        for name, param in self.actor_local.named_parameters():
            if name in grads_actor_clone:
                param.grad = grads_actor_clone[name].clone()

        for name, param in self.critic_local.named_parameters():
            if name in grads_critic_clone:
                param.grad = grads_critic_clone[name].clone()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def update_special_part_withmask(self,original_actor_local,original_critic_local,id_mark,states, states_next,
                                     states_total, states_total_next, actions, rewards, dones,idx_mask,mask_weight):

        self.actor_local.load_state_dict(original_actor_local)  
        self.critic_local.load_state_dict(original_critic_local)
        row_actor = self.actor_masks[idx_mask][0]
        col_actor = self.actor_masks[idx_mask][1]
        row_critic = self.critic_masks[idx_mask][0]
        col_critic = self.critic_masks[idx_mask][1]
        states0, states_next0, states_total0, states_total_next0, actions0, rewards0, dones0 = \
            self.extract_data_train2(states, states_next, states_total, states_total_next, actions, rewards, dones,
                                    idx_mask, mask_weight)
        
        if len(states_total_next0) == 0:  
            with torch.no_grad():
                grad_critic_local2 = self.critic_local.fcs1.weight.grad.clone()
                grad_critic_local2[:, :] = 0
            with torch.no_grad():
                grad_actor_local2 = self.actor_local.fc1.weight.grad.clone()
                grad_actor_local2[:, :] = 0
            return grad_actor_local2,grad_critic_local2

        actions_next0 = self.actor_target(states_total_next0[:, 0:self.state_size])
        Q_targets_next0 = self.critic_target(states_total_next0, actions_next0)
        Q_targets0 = rewards0 + (GAMMA * Q_targets_next0 * (1 - dones0))
        Q_expected0 = self.critic_local(states_total0, actions0)
        critic_loss = F.mse_loss(Q_expected0, Q_targets0)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        with torch.no_grad():
            grad_critic_local2 =  self.critic_local.fcs1.weight.grad.clone()
            grad_critic_local2[:,:] = 0
            grad_critic_local2[row_critic, col_critic] = self.critic_local.fcs1.weight.grad.clone()[row_critic, col_critic]
        actions_pred0 = self.actor_local(states_total0[:, 0:self.state_size])
        actor_loss = -self.critic_local(states_total0, actions_pred0).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        with torch.no_grad():
            grad_actor_local2 =  self.actor_local.fc1.weight.grad.clone()
            grad_actor_local2[:,:] = 0
            grad_actor_local2[row_actor, col_actor] = self.actor_local.fc1.weight.grad.clone()[row_actor, col_actor]
        return grad_actor_local2,grad_critic_local2

    def get_freeze_sharing(self):
        
        
        
        row_freeze_all_actor = []
        col_freeze_all_actor = []
        row_freeze_all_critic = []
        col_freeze_all_critic = []
        n_mask = len(self.actor_masks)
        for i in range(n_mask):
            row_freeze_all_actor.extend( self.actor_masks[i][0])
            col_freeze_all_actor.extend(self.actor_masks[i][1])
            row_freeze_all_critic.extend( self.critic_masks[i][0])
            col_freeze_all_critic.extend(self.critic_masks[i][1])
        return row_freeze_all_actor,col_freeze_all_actor,row_freeze_all_critic,col_freeze_all_critic


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent_c():
    """Interacts with and learns from the environment."""

    def __init__(self, id_agent, state_size, state_total_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.state_total_size = state_total_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.id_agent = id_agent
        self.seed = random_seed

        # Actor Network (w/ Target Network)
        self.actor_local = ActorC(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorC(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR_C,weight_decay=WEIGHT_DECAY_C)
        self.actor_masks = []

        # Critic Network (w/ Target Network)
        self.critic_local = CriticC(state_total_size, action_size, random_seed).to(device)
        self.critic_target = CriticC(state_total_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC_C,
                                           weight_decay=WEIGHT_DECAY_C)  
        self.critic_masks = []  

        # Noise process for each agent
        self.noise = OrnsteinUhlenbeckNoise((num_agents, action_size), random_seed)
        self.noise_scale = 0.1

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def initial_state_dict(self):
        
        self.dict_actor = self.actor_target.state_dict()
        self.dict_critic = self.critic_target.state_dict()

    def get_mask_actor(self, model, percent, list_masked):
        weight = model.fc1.weight.data
        total_params = weight.numel()
        all_indices = torch.arange(total_params)

        num_to_freeze = int(percent * 0.01 * total_params)
        if num_to_freeze == 0:
            mask = [[], []]
            return mask, list_masked

        available_indices = torch.tensor(
            list(set(all_indices.numpy()) - set(list_masked.numpy()))
        )

        
        available_weights = weight.flatten()[available_indices]
        sorted_order = torch.argsort(available_weights.abs(), descending=True)
        selected = sorted_order[:num_to_freeze]
        random_indices = available_indices[selected]

        rows, cols = np.unravel_index(random_indices.numpy(), weight.shape)
        mask = [list(rows), list(cols)]
        list_masked = torch.cat((list_masked, random_indices), dim=0)
        return mask, list_masked

    def get_mask_critic(self, model, percent, list_masked):
        weight = model.fcs1.weight.data
        total_params = weight.numel()
        all_indices = torch.arange(total_params)

        num_to_freeze = int(percent * 0.01 * total_params)
        if num_to_freeze == 0:
            mask = [[], []]
            return mask, list_masked

        available_indices = torch.tensor(
            list(set(all_indices.numpy()) - set(list_masked.numpy()))
        )

        
        available_weights = weight.flatten()[available_indices]
        sorted_order = torch.argsort(available_weights.abs(), descending=True)
        selected = sorted_order[:num_to_freeze]
        random_indices = available_indices[selected]

        rows, cols = np.unravel_index(random_indices.numpy(), weight.shape)
        mask = [list(rows), list(cols)]
        list_masked = torch.cat((list_masked, random_indices), dim=0)
        return mask, list_masked


    def make_prune_masks2(self, n_mask, percent,list_actor_masked,list_critic_masked):
        

        for i in range(n_mask):
            
            mask_actor_prune,list_actor_masked= self.get_mask_actor(self.actor_target, percent, list_actor_masked)
            mask_critic_prune,list_critic_masked = self.get_mask_critic(self.critic_target, percent, list_critic_masked)

            self.actor_masks.append(mask_actor_prune)
            self.critic_masks.append(mask_critic_prune)

        
        actor_freeze = [ [] for _ in range(n_mask) ]
        critic_freeze = [[] for _ in range(n_mask)]
        for i in range(n_mask):
            listi = list(range(0,n_mask))
            # --> actor
            freezi = [[],[]]
            if i in listi:
                listi.remove(i)
            for j in listi:
                actor_i = self.actor_masks[j]
                freezi[0].extend(actor_i[0])
                freezi[1].extend(actor_i[1])
            actor_freeze[i] = freezi

            # --> critic
            freezi = [[],[]]
            for j in listi:
                critic_i = self.critic_masks[j]
                freezi[0].extend(critic_i[0])
                freezi[1].extend(critic_i[1])
            critic_freeze[i] = freezi
        self.actor_freeze = actor_freeze
        self.critic_freeze = critic_freeze

    def step_mask3(self, state, state_next, state_total, state_total_next, action, reward, done,
            id_mask,weight_ls):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # state, state_next, state_total, state_total_next, action, action_other, reward, done
        self.memory.add(state, state_next, state_total, state_total_next, action, reward, done,
                        1, 1, 1,weight_ls, id_mask)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            # print("Learning")
            experiences = self.memory.sample()
            self.learn_mask3(experiences)

    def act_mask2(self, state, idx_mask, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        actor_freeze = self.actor_freeze[idx_mask]
        row_freeze = actor_freeze[0]
        col_freeze = actor_freeze[1]

        # ================= DEBUG =================
        debug_prune = False
        if random.random() < 0.001:
            debug_prune = True

        # if debug_prune:
        #     print("\n========== ACT MASK DEBUG ==========")
        #     print("idx_mask:", idx_mask)
        #     print("num frozen weights:", len(row_freeze))
        #     print("state shape:", state.shape)
        # =========================================

        original_values = self.actor_local.fc1.weight[row_freeze, col_freeze].clone()

        acts = np.zeros((self.num_agents, self.action_size))

        self.actor_local.eval()

        with torch.no_grad():
            
            if debug_prune:
                if self.num_agents > 1:
                    action_no_mask = self.actor_local(state[0, :]).cpu().data.numpy()
                else:
                    action_no_mask = self.actor_local(state).cpu().data.numpy()

            
            self.actor_local.fc1.weight[row_freeze, col_freeze] = 0

            if self.num_agents > 1:
                for agent in range(self.num_agents):
                    acts[agent, :] = self.actor_local(state[agent, :]).cpu().data.numpy()
            else:
                acts[0, :] = self.actor_local(state).cpu().data.numpy()

            # if debug_prune:
            #     action_masked = acts.copy()
            #     print("action_no_mask:", action_no_mask)
            #     print("action_masked:", action_masked)
            #     print("action_diff:", np.abs(action_masked - action_no_mask).mean())
            #     print("====================================\n")

        self.actor_local.train()

        if add_noise:
            noise = self.noise.sample()
            noise = self.noise_scale * noise
            noise_clip = getattr(self, "vx_noise_clip", 0.02)
            noise = np.clip(noise, -noise_clip, noise_clip)

            # if debug_prune:
            #     print("raw noise:", noise / self.noise_scale)
            #     print("scaled noise:", noise)
            #     print("scaled_noise_abs_mean:", np.abs(noise).mean())
            #     print("mask_action_diff:", np.abs(action_masked - action_no_mask).mean())

            acts += noise

        with torch.no_grad():
            self.actor_local.fc1.weight[row_freeze, col_freeze] = original_values

        return np.clip(acts, -1, 1)

    def reset(self):
        self.noise.reset()
    def extract_data_train2(self, states, states_next, states_total, states_total_next, actions, rewards, dones, idx_mask, mask_weight):

        def assign_to_class(probabilities,idx_mask):
            if probabilities[idx_mask] == max(probabilities):  
                return idx_mask
            else:
                return np.random.choice(len(probabilities), p=probabilities)  

        assigned_classes = [assign_to_class(probs,idx_mask) for probs in mask_weight]
        indices = [index for index, value in enumerate(assigned_classes) if value == idx_mask]

        
        states0 = extract_by_indices_tensor(states, indices)
        states_next0 = extract_by_indices_tensor(states_next, indices)
        states_total0 = extract_by_indices_tensor(states_total, indices)
        states_total_next0 = extract_by_indices_tensor(states_total_next, indices)
        actions0 = extract_by_indices_tensor(actions, indices)
        rewards0 = extract_by_indices_tensor(rewards, indices)
        dones0 = extract_by_indices_tensor(dones, indices)

        return states0, states_next0, states_total0, states_total_next0, actions0, rewards0, dones0

    def learn_mask3(self, experiences):
        
        original_actor_local = copy.deepcopy(self.actor_local.state_dict())
        original_critic_local = copy.deepcopy(self.critic_local.state_dict())

        
        states, states_next, states_total, states_total_next, actions, rewards, dones, _, _, _, mask_weight, id_mark = experiences
        row_freeze_all_actor, col_freeze_all_actor, row_freeze_all_critic, col_freeze_all_critic = self.get_freeze_sharing()

        
        actions_next = self.actor_target(states_total_next[:, 0:self.state_size])
        Q_targets_next = self.critic_target(states_total_next, actions_next)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states_total, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        with torch.no_grad():
            grad_critic_local1 =  self.critic_local.fcs1.weight.grad.clone()  
            grad_critic_local1[row_freeze_all_critic, col_freeze_all_critic ] = 0
            self.critic_local.fcs1.weight.grad = grad_critic_local1
        
        grads_critic_clone = {}
        for name, param in self.critic_local.named_parameters():
            if param.grad is not None:  
                grads_critic_clone[name] = param.grad.clone()

        
        actions_pred = self.actor_local(states_total[:, 0:self.state_size])
        actor_loss = -self.critic_local(states_total, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        with torch.no_grad():
            grad_actor_local1 =  self.actor_local.fc1.weight.grad.clone()  
            grad_actor_local1[row_freeze_all_actor, col_freeze_all_actor ] = 0
            self.actor_local.fc1.weight.grad = grad_actor_local1
        
        grads_actor_clone = {}
        for name, param in self.actor_local.named_parameters():
            if param.grad is not None:  
                grads_actor_clone[name] = param.grad.clone()


        
        n_group = len(mask_weight[0])
        for mask_i in range(n_group):
            grad_actor_local_i,grad_critic_local_i = self.update_special_part_withmask(original_actor_local, original_critic_local, id_mark, states,
                                         states_next,states_total, states_total_next, actions, rewards, dones, mask_i,mask_weight)
            
            grads_actor_clone['fc1.weight'] = grads_actor_clone['fc1.weight'] + grad_actor_local_i
            grads_critic_clone['fcs1.weight'] = grads_critic_clone[
                                                    'fcs1.weight'] + grad_critic_local_i

        
        for name, param in self.actor_local.named_parameters():
            if name in grads_actor_clone:
                param.grad = grads_actor_clone[name].clone()

        for name, param in self.critic_local.named_parameters():
            if name in grads_critic_clone:
                param.grad = grads_critic_clone[name].clone()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def update_special_part_withmask(self,original_actor_local,original_critic_local,id_mark,states, states_next,
                                     states_total, states_total_next, actions, rewards, dones,idx_mask,mask_weight):

        self.actor_local.load_state_dict(original_actor_local)  
        self.critic_local.load_state_dict(original_critic_local)
        row_actor = self.actor_masks[idx_mask][0]
        col_actor = self.actor_masks[idx_mask][1]
        row_critic = self.critic_masks[idx_mask][0]
        col_critic = self.critic_masks[idx_mask][1]
        states0, states_next0, states_total0, states_total_next0, actions0, rewards0, dones0 = \
            self.extract_data_train2(states, states_next, states_total, states_total_next, actions, rewards, dones,
                                    idx_mask, mask_weight)

        if len(states_total_next0) == 0:
            with torch.no_grad():
                grad_critic_local2 = self.critic_local.fcs1.weight.grad.clone()
                grad_critic_local2[:, :] = 0
            with torch.no_grad():
                grad_actor_local2 = self.actor_local.fc1.weight.grad.clone()
                grad_actor_local2[:, :] = 0
            return grad_actor_local2, grad_critic_local2


        actions_next0 = self.actor_target(states_total_next0[:, 0:self.state_size])
        Q_targets_next0 = self.critic_target(states_total_next0, actions_next0)
        Q_targets0 = rewards0 + (GAMMA * Q_targets_next0 * (1 - dones0))
        Q_expected0 = self.critic_local(states_total0, actions0)
        critic_loss = F.mse_loss(Q_expected0, Q_targets0)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        with torch.no_grad():
            grad_critic_local2 =  self.critic_local.fcs1.weight.grad.clone()
            grad_critic_local2[:,:] = 0
            grad_critic_local2[row_critic, col_critic] = self.critic_local.fcs1.weight.grad.clone()[row_critic, col_critic]
        actions_pred0 = self.actor_local(states_total0[:, 0:self.state_size])
        actor_loss = -self.critic_local(states_total0, actions_pred0).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        with torch.no_grad():
            grad_actor_local2 =  self.actor_local.fc1.weight.grad.clone()
            grad_actor_local2[:,:] = 0
            grad_actor_local2[row_actor, col_actor] = self.actor_local.fc1.weight.grad.clone()[row_actor, col_actor]
        return grad_actor_local2,grad_critic_local2


    def get_freeze_sharing(self):
        
        
        
        row_freeze_all_actor = []
        col_freeze_all_actor = []
        row_freeze_all_critic = []
        col_freeze_all_critic = []
        n_mask = len(self.actor_masks)
        for i in range(n_mask):
            row_freeze_all_actor.extend( self.actor_masks[i][0])
            col_freeze_all_actor.extend(self.actor_masks[i][1])
            row_freeze_all_critic.extend( self.critic_masks[i][0])
            col_freeze_all_critic.extend(self.critic_masks[i][1])
        return row_freeze_all_actor,col_freeze_all_actor,row_freeze_all_critic,col_freeze_all_critic


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def extract_by_indices_tensor(original_list, indices):
    extracted_elements = [ original_list[i].tolist() for i in indices if i < len(original_list) ]
    extracted_elements_tensor = torch.tensor(extracted_elements)
    return extracted_elements_tensor
