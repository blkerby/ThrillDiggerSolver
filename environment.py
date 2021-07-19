import torch
import torch.nn.functional as F

EMPTY = 0
GREEN = 1
BLUE = 2
RED = 3
SILVER = 4
GOLD = 5
RUPOOR = 6
BOMB = 7

class ThrillDiggerEnvironment:
    def __init__(self, num_envs, size_x, size_y, num_bombs, num_rupoors):
        self.num_envs = num_envs
        self.size_x = size_x
        self.size_y = size_y
        self.num_rupoors = num_rupoors
        self.num_bombs = num_bombs
        self.num_actions = size_x * size_y
        self.num_rupees = size_x * size_y - num_rupoors - num_bombs
        self.num_turns = size_x * size_y - num_bombs
        self.true_map = torch.zeros([num_envs, size_x * size_y], dtype=torch.int8)
        self.observed_map = torch.zeros([num_envs, size_x * size_y], dtype=torch.int8)
        self.done = torch.zeros([num_envs], dtype=torch.bool)
        self.reset()
        self.step_number = 0

    def reset(self, env_ids=None):
        """Reset the given environments to an initial, newly randomized state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        num_envs = env_ids.shape[0]

        # Determine the locations of bombs/rupoors
        bomb_ind = torch.multinomial(torch.ones([num_envs, self.size_x * self.size_y]),
                                       num_samples=self.num_bombs + self.num_rupoors)
        bomb_x = bomb_ind % self.size_x
        bomb_y = bomb_ind // self.size_x
        bomb_map = torch.zeros([num_envs, self.size_x, self.size_y], dtype=torch.int8)
        bomb_map[torch.arange(num_envs).view(-1, 1), bomb_x, bomb_y] = 1

        # Determine the number of neighboring bombs/rupoors for each cell
        cnt_map = F.conv2d(bomb_map.unsqueeze(1),
                           weight=torch.ones([1, 1, 3, 3], dtype=torch.int8),
                           padding=1).squeeze(1)

        # Update `bomb_map` to distinguish rupoors from bombs
        bomb_map[torch.arange(num_envs).view(-1, 1), bomb_x[:, :self.num_rupoors], bomb_y[:, :self.num_rupoors]] = 2

        # Create the "true" map
        true_map = torch.full([num_envs, self.size_x, self.size_y], GREEN, dtype=torch.int8)
        true_map[(cnt_map == 1) | (cnt_map == 2)] = BLUE
        true_map[(cnt_map == 3) | (cnt_map == 4)] = RED
        true_map[(cnt_map == 5) | (cnt_map == 6)] = SILVER
        true_map[(cnt_map == 7) | (cnt_map == 8)] = GOLD
        true_map[bomb_map == 1] = BOMB
        true_map[bomb_map == 2] = RUPOOR

        # Update the corresponding rows in the environment state
        self.true_map[env_ids] = true_map.view(num_envs, -1)
        self.observed_map[env_ids] = 0
        self.done[env_ids] = False
        self.step_number = 0

    def reward(self):
        cnt_green = torch.sum(self.observed_map == GREEN, dim=1)
        cnt_blue = torch.sum(self.observed_map == BLUE, dim=1)
        cnt_red = torch.sum(self.observed_map == RED, dim=1)
        cnt_silver = torch.sum(self.observed_map == SILVER, dim=1)
        cnt_gold = torch.sum(self.observed_map == GOLD, dim=1)
        cnt_rupoor = torch.sum(self.observed_map == RUPOOR, dim=1)
        return torch.clamp_min(cnt_green + 5 * cnt_blue + 20 * cnt_red + 100 * cnt_silver + 300 * cnt_gold - 10 * cnt_rupoor, 0).to(torch.float32)

    def step(self, action):
        # For environments in which the game is not done, reveal the selected cell
        not_done = ~self.done
        action_not_done = action[not_done]
        self.observed_map[not_done, action_not_done] = self.true_map[not_done, action_not_done]

        # For each environment, determine if this move ended the game
        hit_bomb = (self.true_map[torch.arange(self.num_envs, device=action.device), action] == BOMB)
        rupees_collected = torch.sum((self.observed_map >= GREEN) & (self.observed_map <= GOLD), dim=1)
        all_rupees_collected = (rupees_collected == self.num_rupees)
        self.done |= hit_bomb | all_rupees_collected

        self.step_number += 1
# torch.manual_seed(0)
td = ThrillDiggerEnvironment(num_envs=2, size_x=5, size_y=4, num_rupoors=3, num_bombs=1)
self = td
env_ids = torch.arange(self.num_envs)
# td.observed_map = td.true_map
td.reward()
action = torch.tensor([2, 4])
td.step(action)
