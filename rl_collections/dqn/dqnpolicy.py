import os
import pickle
import random
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from collections import abc
from pathlib import Path
from datetime import datetime
from collections import deque
from qnetwork import QNetwork

envname = 'PongNoFrameskip-v4'
saved_policies_maxlen = 10

logging.basicConfig(filename="DQN-{}.log".format(datetime.now().strftime("%Y-%m-%dT%H-%M-%S")),
                    level=logging.INFO)


class DQNPolicy:
    def __init__(self, env_name=envname, device=None, *, no_train: bool=False, active_model_path=None):
        self.device = (device if device is not None
                       else "cuda"
        if torch.cuda.is_available()
        else "cpu")
        print(f"Using device: {self.device}")

        self.env = gym.make(env_name)
        self.env = AtariPreprocessing(self.env, grayscale_obs=True,
                                      scale_obs=True,
                                      terminal_on_life_loss=True)
        self.env = FrameStack(self.env, num_stack=4)
        num_actions: int = self.env.action_space.n

        
        if not no_train:
            self.active_model = QNetwork(num_actions, device=self.device)
            self.active_model.to(self.device)
            self.target_model = QNetwork(num_actions, device=self.device)
            self.target_model.to(self.device)
        else:
            self.active_model = torch.load(active_model_path)
            self.active_model.to(self.device)
            self.target_model = None

        self.num_actions = num_actions
        self.epsilon = 1  # 1 -> 0.01 over 100_000 frames
        self.replay_memory_maxlen = 100_000
        self.replay_memory = deque(maxlen=self.replay_memory_maxlen)
        self.frameskip = 4

        self.framebuffer = deque(maxlen=4)
        self.framectr = 0
        self.minibatch_size = 32
        self.discount_factor = 0.99
        self.current_state = None
        self.model_update_freq = 10000

    def reset_env(self):
        first_frame, _ = self.env.reset()
        first_frame = np.array(first_frame, dtype=np.float32)
        first_frame = torch.from_numpy(first_frame)
        self.current_state = first_frame
        

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
        self.framectr += 1
        self.epsilon = max(0.01, 1 - (1.1e-6 * self.framectr))
        if enable_epsilon and random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.active_model.get_action(self.current_state)
        return action

    def train(self, num_episodes: int):
        Path("models/").mkdir(parents=True, exist_ok=True)
        running_rewards = deque(maxlen=20)
        saved_policies = deque(maxlen=saved_policies_maxlen)
        epsd_reward_list = []
        epsd_reward_list = []
        running_rewards_mean = []
        reward_sum = 0
        logging.info(f"{datetime.now()} - Beginning training for atleast {num_episodes} episodes.")
        for eps_num in range(1, num_episodes + 1):
            epsd_loss, epsd_reward = self.episode()
            running_rewards.append(epsd_reward)
            mean_reward = sum(running_rewards) / len(running_rewards)
            running_rewards_mean.append(mean_reward)
            reward_sum += epsd_reward
            epsd_reward_list.append(epsd_reward)
            epsd_reward_list.append(epsd_reward)

            if mean_reward >= 19.0:  # Save latest model if we hit 19.0 average reward; and exit
                logging.info(
                    f"{datetime.now()} - Episode {eps_num}/{num_episodes}; Epsilon: {self.epsilon:.4f};  Loss: {epsd_loss:.4f}; Reward: {mean_reward}")
                self.save(saved_policies=saved_policies)
                break
            if eps_num % 25 == 0:  # Check progress after every 10th episode
                logging.info(
                    f"{datetime.now()} - Episode {eps_num}/{num_episodes}; Epsilon: {self.epsilon:.4f};  Loss: {epsd_loss:.4f}; Reward: {mean_reward}")
            if eps_num % 25 == 0:  # Save after every 40th episode
                self.save(saved_policies=saved_policies)
            if eps_num == num_episodes:  # Display score and save latest model after final episode of training
                logging.info(
                    f"{datetime.now()} - Episode {eps_num}/{num_episodes}; Epsilon: {self.epsilon:.4f};  Loss: {epsd_loss:.4f}; Reward: {mean_reward}")
                self.save(saved_policies=saved_policies)
        
        DQNPolicy.plot_rewards(running_rewards_mean, epsd_reward_list)
        logging.info(f"{datetime.now()} - Training complete")
        self.env.close()

    def episode(self):
        """
        Runs one episode of training
        :return: epsd_loss, epsd_reward: Loss and reward for that episode.
        """
        # Initialize the sequence
        self.reset_env()

        self.reset_env()

        terminated = False
        epsd_loss = []
        epsd_reward = 0
        while not terminated:
            action = self.get_action(enable_epsilon=True)

            # Execute action(t) in emulator and observe reward(t) and observation(t+1)
            next_state, running_reward, terminated, truncated, info = self.env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            next_state = torch.from_numpy(next_state)
            terminated = terminated or truncated

            epsd_reward += running_reward
            # Store (sequence(t), action(t), sequence(t+1), reward(t), terminated(t))
            experience_tuple = (self.current_state, action, running_reward, next_state, terminated)
            self.current_state = next_state
            self.replay_memory.append(experience_tuple)

            # Sample random mini-batches of experience_tuples from replay_memory
            if len(self.replay_memory) >= self.minibatch_size:
                idx = np.random.choice(len(self.replay_memory), self.minibatch_size)
                exp_minibatch = [self.replay_memory[i] for i in idx]
                loss = self._train_step(exp_minibatch)
                epsd_loss.append(loss)

            # Regularly updating the target model
            if self.framectr % self.model_update_freq == 0:
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
        state_mb = state_mb.to(self.device)
        actions_mb = torch.as_tensor([that_action for (_, that_action, _, _, _) in exp_minibatch])
        actions_mb = actions_mb.to(self.device)
        rewards_mb = torch.as_tensor([that_reward for (_, _, that_reward, _, _) in exp_minibatch], dtype=torch.float32)
        rewards_mb = rewards_mb.to(self.device)
        next_state_mb = torch.stack([that_next_state for (_, _, _, that_next_state, _) in exp_minibatch],
                                    dim=0).to(self.device)
        terminated_mb = torch.as_tensor([if_terminated for (_, _, _, _, if_terminated) in exp_minibatch],
                                        dtype=torch.int).to(self.device)

        # Calculate the actual Q-values for that state and the chosen action
        actions_mb = actions_mb.unsqueeze(dim=1)
        masked_qvals = self.active_model.get_qvalues(state_mb).gather(dim=1, index=actions_mb).squeeze()
        if len(masked_qvals.shape) == 0:  # Little hack to prevent broadcasting errors during loss calculation
            masked_qvals = masked_qvals.unsqueeze(dim=0)

        # Calculate the target Q-values for the next_state
        next_qvals = self.active_model.get_qvalues(next_state_mb)
        # keepdims is set True, because gather needs same dims input as index
        next_qvals_argmax = next_qvals.argmax(dim=1, keepdims=True)
        masked_next_qvals = self.target_model.get_qvalues(next_state_mb).gather(dim=1, index=next_qvals_argmax)
        masked_next_qvals = masked_next_qvals.squeeze()
        if len(masked_next_qvals.shape) == 0:
            masked_next_qvals = masked_next_qvals.unsqueeze(dim=0)
        target = rewards_mb + (1.0 - terminated_mb) * self.discount_factor * masked_next_qvals

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
                fname_ = saved_policies.popleft()
                #os.remove(fname_)
                os.remove(fname_+'.pth')

        if fname is None:
            datetime_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            fname = f"models/policy-{datetime_now}.pkl"

        '''
        with open(fname, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        '''
        torch.save(self.active_model.state_dict(), fname + '.pth')

        if isinstance(saved_policies, deque):
            saved_policies.append(fname)

        #logging.info(f"{datetime.now()} - Saved policy: {fname}")
        logging.info(f"{datetime.now()} - Saved model: {fname}.pth")

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
        model = torch.load(fname+".model")
        policy.active_model = model
        policy.active_model.to(policy.device)

        policy.env = gym.make(envname)

        return policy

    @classmethod
    def plot_rewards(cls, running_rewards, epsd_reward_list):
       X = np.arange(0, len(running_rewards), 1)
       plt.plot(X, epsd_reward_list, color='blue', label='Episode reward')
       plt.plot(X, running_rewards, color='orange', label='Mean rewards of the last 20 episodes')
       plt.xlabel("Number of Episodes")
       plt.ylabel("Reward")
       plt.legend()
       plt.savefig('plot.png')

    @classmethod
    def play_policy(cls, model_file: str, render: bool=False):
        """
        Demonstrate a learned policy.
        :param render:
        :param model_file:
        :returns: `None`
        """
        policy = DQNPolicy(envname, no_train=True, active_model_path=model_file)
        policy.reset_env()
        
        terminated = False
        epsd_reward = 0
        while not terminated:
            action = policy.get_action(enable_epsilon=False)

            # Execute action(t) in emulator and observe reward(t) and observation(t+1)
            next_state, running_reward, terminated, truncated, info = policy.env.step(action)
            next_state, running_reward, terminated, truncated, info = policy.env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            next_state = torch.from_numpy(next_state)
            terminated = terminated or truncated
            policy.current_state = next_state
            epsd_reward += running_reward
        if not render:
            print(f"Reward earned during episode : {epsd_reward}")


def main():
    policy = DQNPolicy(env_name=envname)
    policy.train(num_episodes=2000)



if __name__ == '__main__':
    DQNPolicy.play_policy(model_file="models/pong_dueldqn.pth", render=True)
