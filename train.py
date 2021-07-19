import torch
import logging

from model import Model
from model_average import ExponentialAverage
from environment import ThrillDiggerEnvironment, EMPTY


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])


class TrainingSession:
    def __init__(self,
                 env: ThrillDiggerEnvironment,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 ema_beta: float = 0.0):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.rounds = 0
        self.model_ema = ExponentialAverage(model.parameters(), ema_beta)

    def compute_action_probs(self, action_value: torch.Tensor, temperature: float, explore_eps: float):
        action_valid = self.env.observed_map == EMPTY
        adjusted_action_value = torch.where(action_valid, action_value, torch.full_like(action_value, float('-inf')))
        raw_action_probs = torch.softmax(adjusted_action_value * temperature, dim=1)
        explore_probs = torch.full_like(raw_action_probs, explore_eps / self.env.num_actions) + (
                    1 - explore_eps) * raw_action_probs
        action_probs = torch.where(action_valid, explore_probs, raw_action_probs)
        return action_probs

    def generate_round(self, temperature: float, explore_eps: float):
        device = self.env.observed_map.device
        self.env.reset()
        total_action_prob = 0.0
        observed_map_list = []
        done_list = []
        action_list = []
        state_value_list = []
        action_value_list = []

        with self.model_ema.average_parameters(self.model.parameters()):
            for _ in range(self.env.num_turns):
                # Use the model to compute state and action values
                observed_map = self.env.observed_map.clone()
                done = self.env.done.clone()
                with torch.no_grad():
                    state_value, action_value = self.model(observed_map)

                # Choose actions, with higher-valued actions being more likely
                action_probs = self.compute_action_probs(action_value, temperature, explore_eps)
                action_index = torch.multinomial(action_probs, num_samples=1, replacement=True)[:, 0]
                selected_action_value = action_value[torch.arange(self.env.num_envs, device=device), action_index]

                # Update aggregate of action probabilities, as a metric for exploration
                selected_action_prob = action_probs[torch.arange(self.env.num_envs, device=device), action_index]
                total_action_prob += torch.mean(selected_action_prob)

                # Apply the chosen actions to the environments
                env.step(action_index)

                # Store the relevant data
                observed_map_list.append(observed_map)
                done_list.append(done)
                action_list.append(action_index)
                state_value_list.append(state_value)
                action_value_list.append(selected_action_value)

        reward = env.reward()

        # Stack the data into combined tensors
        observed_map_tensor = torch.stack(observed_map_list, dim=0)
        done_tensor = torch.stack(done_list, dim=0)
        state_value_tensor = torch.stack(state_value_list, dim=0)
        action_value_tensor = torch.stack(action_value_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        return observed_map_tensor, done_tensor, state_value_tensor, action_value_tensor, action_tensor, reward, \
               total_action_prob / self.env.num_turns

    def train_round(self,
                    batch_size: int,
                    temperature: float,
                    explore_eps: float = 0.0,
                    action_loss_weight: float = 0.5,
                    td_lambda: float = 0.0,
                    ):
        device = session.env.observed_map.device

        observed_map, done, state_value, action_value, action, reward, prob = session.generate_round(
            temperature=temperature, explore_eps=explore_eps)

        # Compute eval metrics: reward and Monte-Carlo errors
        mean_reward = torch.mean(reward.to(torch.float32))
        mc_state_err = torch.mean((state_value - reward.unsqueeze(0)) ** 2).item()
        mc_action_err = torch.mean((action_value - reward.unsqueeze(0)) ** 2).item()

        # Compute the TD targets
        target_list = []
        target_batch = reward
        target_list.append(reward)
        for i in reversed(range(1, self.env.num_turns)):
            state_value1 = torch.where(done[i, :], reward, state_value[i, :])
            target_batch = td_lambda * target_batch + (1 - td_lambda) * state_value1
            target_list.append(target_batch)
        target = torch.stack(list(reversed(target_list)), dim=0)

        # Flatten the data
        n = self.env.num_turns * self.env.num_envs
        observed_map = observed_map.view(n, self.env.size_x * self.env.size_y)
        action = action.view(n)
        target = target.view(n)

        # Shuffle the data
        perm = torch.randperm(n)
        observed_map = observed_map[perm, :]
        action = action[perm]
        target = target[perm]

        num_batches = n // batch_size

        total_state_loss = 0.0
        total_action_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            observed_map_batch = observed_map[start:end, :]
            action_batch = action[start:end]
            target_batch = target[start:end]

            state_value_batch, action_value_batch = self.model(observed_map_batch)
            selected_action_value_batch = action_value_batch[torch.arange(batch_size, device=device), action_batch]

            state_loss = torch.mean((state_value_batch - target_batch) ** 2)
            action_loss = torch.mean((selected_action_value_batch - target_batch) ** 2)
            loss = (1 - action_loss_weight) * state_loss + action_loss_weight * action_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-5)
            self.optimizer.step()
            self.model_ema.update(self.model.parameters())
            total_state_loss += state_loss.item()
            total_action_loss += action_loss.item()

        self.rounds += 1
        return mean_reward, mc_state_err, mc_action_err, total_state_loss / num_batches, total_action_loss / num_batches, prob

def beginner_environment(num_envs):
    return ThrillDiggerEnvironment(num_envs, size_x=5, size_y=4, num_bombs=4, num_rupoors=0)

# num_envs = 1
num_envs = 2 ** 18
batch_size = 2 ** 12
ema_beta = 0.98
env = beginner_environment(num_envs=num_envs)
model = Model(env.size_x, env.size_y, hidden_widths=[256, 256])
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999))
session = TrainingSession(env, model, optimizer, ema_beta=ema_beta)
Y = model(env.observed_map)

print(session.model)
print(session.optimizer)
self = session
# temperature = 1.0  # 1e-5
lr0 = 0.002
explore_eps = 0.0
action_loss_weight = 0.5
# temperature = 1000.0

logging.info("Starting training")
for i in range(1000):
    temperature = session.rounds * 0.2 + 1e-5
    optimizer.param_groups[0]['lr'] = lr0 / (session.rounds + 1) ** 0.5
    # observed_map, state_value, action, reward, prob = session.generate_round(
    #     temperature=temperature, explore_eps=explore_eps
    # )
    mean_reward, mc_state_err, mc_action_err, state_loss, action_loss, prob = session.train_round(
        batch_size=batch_size,
        temperature=temperature,
        explore_eps=explore_eps,
        action_loss_weight=action_loss_weight,
        td_lambda=0.0,
    )
    logging.info("{}: reward={:.3f}, mc_state={:.3f}, mc_action={:.3f}, state={:.4f}, action={:.4f}, p={:.4f}".format(
        session.rounds, mean_reward, mc_state_err, mc_action_err, state_loss, action_loss, prob))

# observed_map, state_value, action_value, action, reward, prob = session.generate_round(
#     temperature=temperature, explore_eps=explore_eps)
# i = 4
# print(reward[i])
# print(observed_map[:, i, :])
# print(state_value[:, i])
# print(action_value[:, i])

