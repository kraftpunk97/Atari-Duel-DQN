import os
import pickle
import random
import torch
import logging
import gymnasium as gym
from collections import abc
from pathlib import Path
from datetime import datetime
from collections import deque
from qnetwork import QNetwork
from utils import preprocessing as _preprocessing

envname = 'ALE/Pong-v5'
saved_policies_maxlen = 20


class DQNPolicy:
    def __init__(self, env_name=envname, device=None):
        self.device = (device if device is not None
                       else "cuda"
                       if torch.cuda.is_available()
                       else "cpu")
        print(f"Using device: {self.device}")

        self.env = gym.make(env_name)
        num_actions: int = self.env.action_space.n

        self.active_model = QNetwork(num_actions, device=self.device)
        self.active_model.to(self.device)
        self.target_model = QNetwork(num_actions, device=self.device)
        self.target_model.to(self.device)

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

    def get_qvalues(self, state=None) -> torch.Tensor:
        """
        Get q-values for the current state of the environment

        :param state: torch.Tensor
        :return q_values: torch.Tensor
        """
        state = self.current_state if state is None else state
        return self.active_model(state)

    def get_action(self, enable_epsilon: bool = False):
        """
        Implement epsilon-greedy policy with linear annealing of epsilon
        :param enable_epsilon: `bool` - Whether to implement epsilon-greedy policy
        :return: action
        """
        self.epsilon = max(0.1, 1 - (99e-8 * self.framectr))
        if enable_epsilon and random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.active_model.get_action(self.current_state)
        return action

    def train(self, num_episodes: int):
        logging.basicConfig(filename="DQN-{}.log".format(datetime.now().strftime("%Y-%m-%dT%H-%M-%S")),
                            level=logging.INFO)
        Path("models/").mkdir(parents=True, exist_ok=True)
        running_rewards = deque(maxlen=25)
        saved_policies = deque(maxlen=saved_policies_maxlen)
        for eps_num in range(num_episodes):
            epsd_loss, epsd_reward = self.episode()
            running_rewards.append(epsd_reward)

            if eps_num % 25 == 0:
                mean_reward = sum(running_rewards) / len(running_rewards)
                logging.info(f"{datetime.now()} - Episode {eps_num} loss: {epsd_loss}; Reward: {mean_reward}")
                self.save(saved_policies=saved_policies)
        logging.info(f"{datetime.now()} - Training complete")
        self.env.close()

    def episode(self):
        """
        Runs one episode of training
        :return: epsd_loss, epsd_reward: Loss and reward for that episode.
        """
        # Initialize the sequence
        first_frame, _ = self.env.reset()
        self.framebuffer.clear()
        for _ in range(4):
            self.framebuffer.append(first_frame)
        preprocessed_first_frame = _preprocessing(self.framebuffer)
        first_state = torch.stack([torch.from_numpy(frame).type(torch.float32)
                                   for frame in preprocessed_first_frame])
        self.current_state = first_state

        terminated = False
        epsd_loss = []
        epsd_reward = 0
        while not terminated:
            action = self.get_action(enable_epsilon=True)

            # Execute action(t) in emulator and observe reward(t) and observation(t+1)
            running_reward = 0
            for _ in range(self.frameskip):
                frame, reward, terminated, truncated, info = self.env.step(action)
                reward = (1
                          if reward > 0.0
                          else -1
                          if reward < 0.0
                          else 0)
                self.framectr += 1
                running_reward += reward

                if terminated or truncated:
                    terminated = True
                    break

            self.framebuffer.append(frame)  # framebuffer is a deque

            # Preprocess sequence(t+1)
            preprocessed_fb = _preprocessing(self.framebuffer)
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
            loss = self._train_step(exp_minibatch)
            epsd_loss.append(loss)

            # Regularly updating the target model
            if self.framectr % 10000 == 0:
                self.target_model.load_state_dict(self.active_model.state_dict())

        epsd_loss = sum(epsd_loss) / len(epsd_loss)
        return epsd_loss, epsd_reward

    def _train_step(self, exp_minibatch: abc.Iterable):
        """
        Calculates the loss from the difference between the actual Q-values and the
        target Q-values.
        :param exp_minibatch:
        :return:
        """
        state_mb = torch.stack([that_state for (that_state, _, _, _, _) in exp_minibatch], dim=0)
        next_state_mb = torch.stack([that_next_state for (_, _, _, that_next_state, _) in exp_minibatch],
                                    dim=0)
        rewards_mb = torch.as_tensor([that_reward for (_, _, that_reward, _, _) in exp_minibatch])
        terminated_mb = torch.as_tensor([if_terminated for (_, _, _, _, if_terminated) in exp_minibatch],
                                        dtype=torch.int)
        actions_mb = torch.as_tensor([that_action for (_, that_action, _, _, _) in exp_minibatch])

        # Calculate the actual Q-values for that state and the chosen action
        actions_mb = actions_mb.unsqueeze(dim=1)
        masked_qvals = self.active_model.get_qvalues(state_mb).gather(dim=1, index=actions_mb).squeeze()

        # Calculate the target Q-values for the next_state
        next_qvals = self.active_model.get_qvalues(next_state_mb)
        # keepdims is set True, because gather needs same dims input as index
        next_qvals_argmax = next_qvals.argmax(dim=1, keepdims=True)
        masked_next_qvals = self.target_model.get_qvalues(next_state_mb).gather(dim=1, index=next_qvals_argmax)
        target = rewards_mb + (1.0-terminated_mb)*self.discount_factor*masked_next_qvals

        # Detaching `target` tensor because we only want to update the weights of active_model.
        loss = self.active_model.train_step(masked_qvals, target.detach())
        return loss

    def save(self, fname=None, saved_policies=None):
        """
        Saves the current policy on a file. Saves policy and the active_model separately.
        :param fname: Name of the save file. If set `None`, then the datetimestamp is used.
        :param saved_policies: (`None`, `collections.deque`) Only `saved_policies.maxlen` recent policies are saved
        :return:
        """
        if isinstance(saved_policies, deque):
            if len(saved_policies) >= saved_policies.maxlen:
                fname = saved_policies.popleft()
                os.remove(fname)
                os.remove(fname+'.active_model')

        if fname is None:
            datetime_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            fname = f"models/policy-{datetime_now}.pkl"

        with open(fname, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        torch.save(self.active_model, fname + '.active_model')

        if isinstance(saved_policies, deque):
            saved_policies.append(fname)

        logging.info(f"{datetime.now()} - Saved policy: {fname}")
        logging.info(f"{datetime.now()} - Saved active_model: {fname}.active_model")

    @classmethod
    def load(cls, fname: str):
        """
        Load a previously saved policy
        :param fname: File name of the policy. Must have a corresponding `.active_model` file as well.
        :return: policy: `DQNPolicy`
        """
        if fname is None:
            raise AttributeError("'fname' argument can not be None")
        with open(fname, 'rb') as inp:
            policy = pickle.load(inp)
        model = torch.load(fname+".active_model")
        policy.active_model = model
        policy.active_model.to(policy.device)

        policy.env = gym.make(envname)

        return policy


def main():
    policy = DQNPolicy


def load_test(fname):
    policy = DQNPolicy.load(fname)


if __name__ == '__main__':
    main()
