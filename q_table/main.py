import numpy as np
import random
import heapq

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros(state_space + (action_space,))
        print(self.q_table.shape)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state[0], state[1], valid_actions]
            return valid_actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state, next_valid_actions):
        if not next_valid_actions:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state[0], next_state[1], next_valid_actions])
            
        current_q = self.q_table[state[0], state[1], action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state[0], state[1], action] = new_q

    def get_valid_actions(self, state, grid):
        actions = []
        rows, cols = len(grid), len(grid[0])
        r, c = state
        
        # Order: Up, Down, Left, Right
        # Check Up
        if r > 0 and grid[r - 1][c] == 0:
            actions.append(0)
        # Check Down
        if r < rows - 1 and grid[r + 1][c] == 0:
            actions.append(1)
        # Check Left
        if c > 0 and grid[r][c - 1] == 0:
            actions.append(2)
        # Check Right
        if c < cols - 1 and grid[r][c + 1] == 0:
            actions.append(3)
            
        return actions

def q_learning_search(start, goal, grid, episodes, max_steps):
    rows, cols = len(grid), len(grid[0])
    agent = QLearningAgent(state_space=(rows, cols), action_space=4)
    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state = start
        total_reward = 0
        steps = 0
        done = False

        while not done:
            valid_actions = agent.get_valid_actions(state, grid)
            action = agent.choose_action(state, valid_actions)

            if action is None: # No valid actions
                break

            if action == 0:
                next_state = (state[0] - 1, state[1])
            elif action == 1:
                next_state = (state[0] + 1, state[1])
            elif action == 2:
                next_state = (state[0], state[1] - 1)
            else: # action == 3
                next_state = (state[0], state[1] + 1)

            if next_state == goal:
                reward = 100
                done = True
            else:
                reward = -1 # Small penalty for each step
            
            total_reward += reward
            
            next_valid_actions = agent.get_valid_actions(next_state, grid)
            agent.update_q_table(state, action, reward, next_state, next_valid_actions)
            
            state = next_state
            steps += 1
            
            if done or steps >= max_steps:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    # Reconstruct path from Q-table
    path = []
    state = start
    visited = set()
    while state != goal and state not in visited:
        visited.add(state)
        path.append(state)
        valid_actions = agent.get_valid_actions(state, grid)
        if not valid_actions:
            path = None  # No path found
            break
        
        q_values = agent.q_table[state[0], state[1], valid_actions]
        action = valid_actions[np.argmax(q_values)]
        
        if action == 0:
            state = (state[0] - 1, state[1])
        elif action == 1:
            state = (state[0] + 1, state[1])
        elif action == 2:
            state = (state[0], state[1] - 1)
        else: # action == 3
            state = (state[0], state[1] + 1)
            
        if len(path) > rows * cols: # Path is too long, something is wrong
            path = None
            break

    if path:
        path.append(goal)
    
    # Fallback to A* if no path is found
    if path is None:
        fallback_path = a_star_search(start, goal, grid)
        return fallback_path, agent, episode_rewards, episode_lengths

    return path, agent, episode_rewards, episode_lengths

def a_star_search(start, goal, grid):
    """A* search algorithm"""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()
    
    while open_set:
        _, cost_so_far, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
            
        if current in visited:
            continue
        visited.add(current)
        
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if (0 <= neighbor[0] < rows and
                0 <= neighbor[1] < cols and
                grid[neighbor[0]][neighbor[1]] == 0 and
                neighbor not in visited):
                new_cost = cost_so_far + 1
                heapq.heappush(open_set, (new_cost + heuristic(neighbor, goal), new_cost, neighbor, path + [neighbor]))
    
    return None

# Test with the same maze as A*
if __name__ == "__main__":
    # 0 = free space, 1 = wall
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]

    start = (0, 0)
    goal = (4, 4)
    
    print("Training Q-learning agent...")
    path, agent, rewards, lengths = q_learning_search(start, goal, maze, episodes=1000, max_steps=1000)
    
    print("Path found:" if path else "No path found.")
    if path:
        print("Path:", path)
        print("Path length:", len(path))
        print(f"Final exploration rate: {agent.epsilon:.3f}")
        print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")