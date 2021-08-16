import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Memory:
    # Memória usada para replays

    def __init__(self):
        self.buffer = {"states":[],
                       "actions":[],
                       "rewards":[],
                       "next_states":[],
                       "dones":[]}
    def __len__(self):
        return len(self.buffer["states"])

    def append(self, experience):
        for exp, locus in zip(experience, self.buffer.values()):
            locus.append(exp)

    def reset(self):
        for value in self.buffer.values():
            value.clear()  

class ActorCritic2Head(nn.Module):
    # Rede neural

    def __init__(self, inshape, outshape):
        super().__init__()

        self.inshape = inshape
        self.outshape = outshape

        self.conv1 = nn.Conv2d(4, 32, 7, 3)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.conv3 = nn.Conv2d(64, 64, 3 ,1)
        self.linear1 = nn.Linear(9*9*64, 512)
        self.actor = nn.Linear(512, self.outshape)
        self.critic = nn.Linear(512, 1)

    def _get_conv_out(self, data):
        x = F.relu(self.conv1(data))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size()[0], -1)
    
    def forward(self, x):
        x = self._get_conv_out(x)
        x = F.relu(self.linear1(x))
        policy = F.softmax(self.actor(x), dim = 1)
        value = self.critic(x)
        return policy, value

class A2C:
    def __init__(self, inshape, outshape,
                 lr, gamma, entropy_coef, batch, num_envs,
                 load = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inshape = inshape
        self.outshape = outshape
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch = batch
        self.num_envs = num_envs
        self.memory = Memory()

        if not load:
            print("MODEL CREATED!")
            self.A2C = ActorCritic2Head(self.inshape, self.outshape).to(self.device)
        else:
            print("MODEL LOADED!")
            self.A2C = ActorCritic2Head(self.inshape, self.outshape).to(self.device)
            self.A2C.load_state_dict(torch.load(load, map_location=self.device))
            
        self.opt = optim.Adam(self.A2C.parameters(),
                                       lr=lr, eps=1e-3)

    def choose_action(self, states, test = False):
        # escolhe a ação

        states = torch.tensor(states).to(self.device, dtype = torch.float)
        if test:
            states = states.unsqueeze(0)
        logits, _ = self.A2C(states)
        probs = torch.distributions.Categorical(logits)
        actions = probs.sample().cpu().detach().numpy()
        return actions

    def compute_losses(self, states, actions, returns, advantages):
        # calcula a loss

        logits, values = self.A2C(states.reshape(-1,4,84,84))
        logits = logits.reshape(-1,self.num_envs,self.outshape)
        values = values.reshape(-1,self.num_envs,)
        probs = torch.distributions.Categorical(logits)
        log_probs = -probs.log_prob(actions)

        policy_loss = (log_probs*advantages.detach()).mean()

        entropy_loss = -self.entropy_coef*probs.entropy().mean()
        value_loss = F.mse_loss(values.double(), returns.detach())
        self.opt.zero_grad()
        (policy_loss + entropy_loss + value_loss).backward()
        self.opt.step()

    def train(self):
        # ciclo de treino 

        if len(self.memory) < self.batch:
            return

        batch = self.memory.buffer

        states = torch.tensor(batch["states"]).to(self.device,dtype = torch.float)
        actions = torch.tensor(batch["actions"]).to(self.device)
        rewards = torch.tensor(batch["rewards"]).to(self.device)
        next_states = torch.tensor(batch["next_states"]).to(self.device,dtype = torch.float)
        dones = torch.tensor(batch["dones"]).to(self.device)

        self.memory.reset()

        _, values = self.A2C(states.reshape(-1,4,84,84))
        _, next_values = self.A2C(next_states.reshape(-1,4,84,84))
        values = values.reshape(-1, self.num_envs,)
        next_values = next_values.reshape(-1, self.num_envs,)

        deltas = rewards + torch.logical_not(dones) * self.gamma * next_values - values

        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)

        returns[-1] = rewards[-1] + self.gamma * torch.logical_not(dones)[-1] * next_values[-1]
        advantages[-1] = deltas[-1]

        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + self.gamma * torch.logical_not(dones)[i] * returns[i + 1]
            advantages[i] = deltas[i] + self.gamma * torch.logical_not(dones)[i] * advantages[i + 1]

        self.compute_losses(states, actions, returns, advantages)
        return       