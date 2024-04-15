from pathlib import Path
import random
import torch
import logging
import gymnasium as gym
from datetime import datetime
from collections import deque
from qnetwork import QNetwork
from utils import preprocessing
#from rl_collections.dqn.qnetwork import QNetwork
#from rl_collections.dqn.utils import preprocessing

logging.basicConfig(filename=f"DQN-{datetime.now()}.log", level=logging.INFO)
Path("models/").mkdir(parents=True, exist_ok=True)

class DQNPolicy:
    def __init__(self, num_actions: int, device=None):
        self.device = (device if device is not None
                       else "cuda"
                       if torch.cuda.is_available()
                       else "cpu")
        print(f"Using device: {self.device}")
        self.model = QNetwork(num_actions, device=self.device)
        self.model.to(self.device)
        self.num_actions = num_actions
        self.epsilon = 1  # 1 -> 0.1 over 1_000_000 frames
        self.replay_memory_maxlen = 1_000_000
        self.replay_memory = deque(maxlen=self.replay_memory_maxlen)
        self.frameskip = 4

        self.framebuffer = deque(maxlen=4)
        self.framectr = 0
        self.minibatch_size = 32
        self.discount_factor = 0.99
        self.current_state = None

        self.env = gym.make('ALE/Pong-v5')

    def get_qvalues(self, state=None) -> torch.Tensor:
        """
        Get q-values for the current state of the environment

        :param state: torch.Tensor
        :return q_values: torch.Tensor
        """
        state = self.current_state if state is None else state
        return self.model(state)

    def episode(self):
        """
        Runs one episode of training
        :return:
        """
        # Initialize the sequence
        first_frame, _ = self.env.reset()
        self.framebuffer.clear()
        for _ in range(4):
            self.framebuffer.append(first_frame)
        preprocessed_first_frame = preprocessing(self.framebuffer)
        first_state = torch.stack([torch.from_numpy(frame).type(torch.float32)
                                   for frame in preprocessed_first_frame])
        self.current_state = first_state

        terminated = False
        epsd_loss = []
        epsd_reward = 0
        while not terminated:

            # Implement epsilon-greedy policy with linear annealing of epsilon
            self.epsilon = max(0.1, 1-(99e-8*self.framectr))
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.model.get_action(self.current_state)

            self.model.get_action(self.current_state)
            # Execute action(t) in emulator and observe reward(t) and observation(t+1)
            running_reward = 0
            for _ in range(self.frameskip):
                frame, reward, terminated, truncated, info = self.env.step(action)
                reward = (1
                          if reward > 0
                          else -1
                          if reward < 0
                          else 0)
                self.framectr += 1
                running_reward += reward

                if terminated or truncated:
                    terminated = True
                    break

            self.framebuffer.append(frame)

            # Preprocess sequence(t+1)
            preprocessed_fb = preprocessing(self.framebuffer)
            next_state = torch.stack([torch.from_numpy(frame).type(torch.float32)
                                      for frame in preprocessed_fb])
            epsd_reward += running_reward
            # Store (sequence(t), action(t), sequence(t+1), reward(t), terminated(t))
            experience_tuple = (self.current_state, action, running_reward, next_state, terminated)
            self.current_state = next_state
            self.replay_memory.append(experience_tuple)


            # Sample random mini-batches of experience_tuples from replay_memory
            minibatch_size = min(self.minibatch_size, len(self.replay_memory))
            exp_minibatch = random.sample(self.replay_memory, minibatch_size)
            y = torch.Tensor([that_reward if if_terminated
                              else that_reward + self.discount_factor*torch.max(self.get_qvalues(that_next_state))
                              for (that_state, that_action, that_reward, that_next_state, if_terminated)
                              in exp_minibatch])
            state_minibatch = torch.stack([that_state for (that_state, _, _, _, _) in exp_minibatch], dim=0)
            action_minibatch = [that_action for (_, that_action, _, _, _) in exp_minibatch]

            # Perform gradient descent step on (y - self.model(self.current_state)[action])^2
            epsd_loss.append(self.model.optimize(minibatch=(state_minibatch, action_minibatch), y=y))

        epsd_loss = sum(epsd_loss) / len(epsd_loss)
        return epsd_loss, epsd_reward


def main():
    policy = DQNPolicy(6)
    for eps_num in range(10000):
        epsd_loss, epsd_reward = policy.episode()
        if eps_num % 10 == 0:
            logging.info(f"{datetime.now()} - Episode {eps_num} loss: {epsd_loss}; Reward: {epsd_reward}")
        if eps_num % 100 == 0:
            torch.save(policy,f"models/{datetime.now()}")
    #print(len(policy.replay_memory))


if __name__ == '__main__':
    main()
