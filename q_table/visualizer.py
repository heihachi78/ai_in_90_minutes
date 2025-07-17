import pygame
import sys
import numpy as np
import threading
from main import q_learning_search

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 30
COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'light_blue': (173, 216, 230),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128)
}

class QLearningVisualizer:
    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = None
        self.goal = None
        self.path = []
        self.mode = 'wall'  # 'wall', 'start', 'goal'
        self.mouse_pressed = False
        self.wall_drawing_mode = None
        
        # Q-learning specific
        self.agent = None
        self.training = False
        self.show_q_values = False
        self.used_fallback = False
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode = 0
        self.training_path = []
        self.q_colors_cache = None
        
        # Set up display
        self.width = cols * CELL_SIZE
        self.height = rows * CELL_SIZE + 150  # Extra space for controls
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-Learning Pathfinding Visualizer")
        
        # Font for text
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        
    def get_cell_from_pos(self, pos):
        x, y = pos
        if y < self.rows * CELL_SIZE:
            col = x // CELL_SIZE
            row = y // CELL_SIZE
            if 0 <= row < self.rows and 0 <= col < self.cols:
                return row, col
        return None
        
    def handle_click(self, pos):
        cell = self.get_cell_from_pos(pos)
        if cell:
            row, col = cell
            
            if self.mode == 'wall':
                if self.wall_drawing_mode is None:
                    self.wall_drawing_mode = 1 - self.grid[row][col]
                self.grid[row][col] = self.wall_drawing_mode
            elif self.mode == 'start':
                if self.start:
                    old_row, old_col = self.start
                    if self.grid[old_row][old_col] == 0:
                        pass
                self.start = (row, col)
                self.grid[row][col] = 0
            elif self.mode == 'goal':
                if self.goal:
                    old_row, old_col = self.goal
                    if self.grid[old_row][old_col] == 0:
                        pass
                self.goal = (row, col)
                self.grid[row][col] = 0
                
    def handle_mouse_motion(self, pos):
        if self.mouse_pressed and self.mode == 'wall':
            cell = self.get_cell_from_pos(pos)
            if cell:
                row, col = cell
                if self.wall_drawing_mode is not None:
                    self.grid[row][col] = self.wall_drawing_mode
                    
    def train_agent_threaded(self, episodes=100000, max_steps=0):
        """Train the Q-learning agent in a separate thread"""
        self.training = True
        self.current_episode = 0
        if max_steps == 0:
            max_steps = self.rows * self.cols
        
        def train():
            try:
                self.path, self.agent, self.episode_rewards, self.episode_lengths = q_learning_search(
                    self.start, self.goal, self.grid, episodes, max_steps
                )
                if self.path is None:
                    self.path = []
                    self.used_fallback = True
                else:
                    # Check if A* fallback was used by seeing if agent learned anything
                    if self.agent and np.max(np.abs(self.agent.q_table)) < 0.1:
                        self.used_fallback = True
                    else:
                        self.used_fallback = False
            except Exception as e:
                print(f"Training error: {e}")
                self.path = []
                self.used_fallback = True
            finally:
                self.training = False
            
        thread = threading.Thread(target=train)
        thread.daemon = True
        thread.start()
        
    def get_q_value_colors(self):
        """Get colors for cells based on state values (expected return from that state)"""
        if self.q_colors_cache is not None:
            return self.q_colors_cache

        if not self.agent:
            return {}
            
        colors = {}
        state_values = {}
        
        # Calculate state values (expected return from each state)
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == 0:  # Free space
                    # State value is the maximum Q-value for valid actions only
                    valid_actions = self.agent.get_valid_actions((row, col), self.grid)
                    if valid_actions:
                        valid_q_values = [self.agent.q_table[row][col][action] for action in valid_actions]
                        state_values[(row, col)] = max(valid_q_values)
                    else:
                        state_values[(row, col)] = 0
        
        if not state_values:
            return {}
            
        # Find min and max state values for normalization
        values = list(state_values.values())
        max_val = max(values)
        min_val = min(values) if values else 0
        
        # Create meaningful color mapping
        for (row, col), value in state_values.items():
            # If all values are very close to zero (unexplored), show neutral colors
            if max_val - min_val < 0.5:
                # Use distance to goal for coloring when Q-values are not learned yet
                distance_to_goal = abs(row - self.goal[0]) + abs(col - self.goal[1]) if self.goal else 0
                max_distance = self.rows + self.cols
                distance_normalized = 1 - (distance_to_goal / max_distance)
                
                # Light blue gradient based on distance to goal
                red = int(200 + 55 * (1 - distance_normalized))
                green = int(200 + 55 * (1 - distance_normalized))
                blue = 255
                colors[(row, col)] = (red, green, blue)
            else:
                # Normal Q-value based coloring
                if max_val > min_val:
                    normalized = (value - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                
                # More intuitive color scheme:
                # Green = high value (good states)
                # Yellow = medium value 
                # Red = low value (bad states)
                if normalized > 0.7:
                    # High value: Green
                    red = int(50 + 50 * (1 - normalized))
                    green = 255
                    blue = int(50 + 50 * (1 - normalized))
                elif normalized > 0.3:
                    # Medium value: Yellow to light green
                    red = int(255 * (1 - (normalized - 0.3) / 0.4))
                    green = 255
                    blue = int(50 + 100 * (normalized - 0.3) / 0.4)
                else:
                    # Low value: Red to yellow
                    red = 255
                    green = int(100 + 155 * (normalized / 0.3))
                    blue = 50
                
                colors[(row, col)] = (red, green, blue)
        
        self.q_colors_cache = colors
        return colors
        
    def draw_grid(self):
        q_colors = self.get_q_value_colors() if self.agent and not self.training and self.show_q_values else {}
        
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * CELL_SIZE
                y = row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                
                # Determine cell color
                if self.grid[row][col] == 1:
                    color = COLORS['black']  # Wall
                elif (row, col) == self.start:
                    color = COLORS['green']  # Start
                elif (row, col) == self.goal:
                    color = COLORS['red']    # Goal
                elif self.path and (row, col) in self.path:
                    color = COLORS['yellow'] # Path
                elif (row, col) in q_colors:
                    color = q_colors[(row, col)]  # Q-value visualization
                else:
                    color = COLORS['white']  # Free space
                    
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS['gray'], rect, 1)
                
    def draw_controls(self):
        y_offset = self.rows * CELL_SIZE + 10
        
        # Mode indicators
        modes = [
            ('W - Wall Mode', self.mode == 'wall'),
            ('S - Start Mode', self.mode == 'start'),
            ('G - Goal Mode', self.mode == 'goal')
        ]
        
        for i, (text, active) in enumerate(modes):
            color = COLORS['blue'] if active else COLORS['black']
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (10, y_offset + i * 22))
            
        # Q-learning controls
        controls = [
            "T - Train Agent",
            f"V - Q-Values: {'ON' if self.show_q_values else 'OFF'}",
            "R - Reset Grid",
            "C - Clear Path"
        ]
        
        if self.show_q_values and self.agent:
            controls.append("Colors: Green=Good, Yellow=OK, Red=Bad")
        
        for i, text in enumerate(controls):
            text_surface = self.font.render(text, True, COLORS['black'])
            self.screen.blit(text_surface, (150, y_offset + i * 22))
            
        # Training status
        if self.training:
            status = "Training..."
            color = COLORS['orange']
        elif self.agent:
            if self.used_fallback:
                status = "Used A* Fallback"
                color = COLORS['purple']
            else:
                status = "Q-Learning Success"
                color = COLORS['green']
        else:
            status = "No Agent"
            color = COLORS['red']
            
        status_surface = self.font.render(status, True, color)
        self.screen.blit(status_surface, (280, y_offset))
        
        # Statistics
        if self.agent and self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-100:])
            stats_text = f"Avg Reward: {avg_reward:.1f}"
            stats_surface = self.small_font.render(stats_text, True, COLORS['black'])
            self.screen.blit(stats_surface, (280, y_offset + 22))
            
            if self.path:
                path_text = f"Path Length: {len(self.path)}"
                path_surface = self.small_font.render(path_text, True, COLORS['black'])
                self.screen.blit(path_surface, (280, y_offset + 44))
            else:
                no_path_text = "No Path Found"
                no_path_surface = self.small_font.render(no_path_text, True, COLORS['red'])
                self.screen.blit(no_path_surface, (280, y_offset + 44))
                
    def reset_grid(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.start = None
        self.goal = None
        self.path = []
        self.agent = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.training = False
        self.show_q_values = False
        self.used_fallback = False
        self.q_colors_cache = None
        
    def clear_path(self):
        self.path = []
        
    def toggle_q_values(self):
        """Toggle between showing Q-values and normal view"""
        if self.agent:
            self.show_q_values = not self.show_q_values
        
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.mouse_pressed = True
                        self.handle_click(event.pos)
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        self.mouse_pressed = False
                        self.wall_drawing_mode = None
                        
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos)
                        
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.mode = 'wall'
                    elif event.key == pygame.K_s:
                        self.mode = 'start'
                    elif event.key == pygame.K_g:
                        self.mode = 'goal'
                    elif event.key == pygame.K_t:
                        if self.start and self.goal and not self.training:
                            self.q_colors_cache = None # Invalidate cache
                            self.train_agent_threaded(episodes=10000)
                    elif event.key == pygame.K_v:
                        self.toggle_q_values()
                    elif event.key == pygame.K_r:
                        self.reset_grid()
                    elif event.key == pygame.K_c:
                        self.clear_path()
                        
            # Draw everything
            self.screen.fill(COLORS['light_blue'])
            self.draw_grid()
            self.draw_controls()
            
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    visualizer = QLearningVisualizer(12, 15)
    visualizer.run()