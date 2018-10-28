from collections import deque, namedtuple
import random

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from sumtree import SumTree
from model import QNetwork

RESULT_DIR = "./results/"
BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, state_size, action_size, use_double_dqn=False,  eps_start=1.0, eps_end=0.001, eps_decay=0.995, seed=13):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            use_double_dqn(bool): use doubl DQN or not
            eps_start: initial value parameter for epsilon greedy
            eps_end: minimum value parameter for epsilon greedy
            eps_decay: decay rate parameter for epsilon greedy
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        self.use_double_dqn = use_double_dqn

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = eps_start

        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.use_double_dqn:
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def set_next_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_end)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def filled_buffer_ratio(self):
        return len(self.memory) / BUFFER_SIZE

    def save_model(self, model_name):
        target_model_path = RESULT_DIR + model_name + "_target_weight.pth"
        local_model_path = RESULT_DIR + model_name + "_local_weight.pth"
        torch.save(self.qnetwork_target.state_dict(), target_model_path)
        torch.save(self.qnetwork_local.state_dict(), local_model_path)

    def load_model(self, model_name):
        target_model_path = RESULT_DIR + model_name + "_target_weight.pth"
        local_model_path = RESULT_DIR + model_name + "_local_weight.pth"
        self.qnetwork_target.load_state_dict(torch.load(target_model_path))
        self.qnetwork_local.load_state_dict(torch.load(local_model_path))


class PrioritizedAgent(Agent):
    show_switched = True

    def __init__(self, state_size, action_size, use_double_dqn=False,  eps_start=1.0, eps_end=0.001, eps_decay=0.995, seed=13):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            use_double_dqn(bool): use doubl DQN or not
            eps_start: initial value parameter for epsilon greedy
            eps_end: minimum value parameter for epsilon greedy
            eps_decay: decay rate parameter for epsilon greedy
            seed (int): random seed
        """
        super().__init__(state_size, action_size, use_double_dqn,  eps_start, eps_end, eps_decay, seed)

        # Prioritized Replay Replay Buffer
        self.p_memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.from_numpy(np.array([action])).long().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                next_action = self.qnetwork_local(next_state).detach().max(1)[1].unsqueeze(1)
                Q_targets_next = self.qnetwork_target(next_state).detach().gather(1, next_action)
            else:
                # Get max predicted Q values (for next states) from target model
                Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states
            Q_target = reward + (GAMMA * Q_targets_next * (1 - done))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(state).gather(1, action)

            td_error = abs(Q_target - Q_expected)
        self.qnetwork_local.train()

        self.memory.add(state, action, reward, next_state, done)
        self.p_memory.add(td_error, state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            filled_buffer_size = len(self.memory)
            if filled_buffer_size < BUFFER_SIZE:
                # Buffer is not filled
                # Use regular memory buffer until buffer is filled.
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, 1, GAMMA)
            else:
                # Buffer is filled, start to use prioritized buffer
                # When buffer is filled, start to use prioritized sampling
                if self.show_switched:
                    print("\nswitched")
                    self.show_switched = False

                idx_exp = self.p_memory.sample()
                indecis, weights, experiences = idx_exp[0], idx_exp[1], idx_exp[2:]
                td_errors = self.learn(experiences, weights, GAMMA)
                self.update_td_error(indecis, td_errors)

    def learn(self, experiences, weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.use_double_dqn:
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = weights * Q_targets

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_expected = weights * Q_expected

        td_errors = np.abs((Q_targets.detach() - Q_expected.detach()).data.numpy()).squeeze()
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        return td_errors

    def update_td_error(self, indecis, td_errors):
        for idx, td_error in zip(indecis, td_errors):
            self.p_memory.update(idx, td_error)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    e = 1e-5
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.tree = SumTree(buffer_size)

    def _get_priority(self, error):
        return (error + self.e) ** self.alpha

    def add(self, error, state, action, reward, next_state, done):
        sample = self.experience(state, action, reward, next_state, done)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self):
        experiences = []
        indecis = []
        segment = self.tree.total() / self.batch_size

        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            if len(data) < 5:
                raise ValueError("missed data")
            priorities.append(p)
            experiences.append(data)
            indecis.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        is_weights = torch.from_numpy(np.vstack([w for w in is_weights if w is not None])).float().to(device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (indecis, is_weights, states, actions, rewards, next_states, dones)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
