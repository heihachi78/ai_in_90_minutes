import pygame
import sys
from main import a_star_search

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 20
CELL_SIZE = 30
COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'light_blue': (173, 216, 230)
}

class AStarVisualizer:
    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = None
        self.goal = None
        self.path = []
        self.mode = 'wall'  # 'wall', 'start', 'goal'
        self.mouse_pressed = False
        self.wall_drawing_mode = None  # Will be set to 0 or 1 when dragging
        
        # Set up display
        self.width = cols * CELL_SIZE
        self.height = rows * CELL_SIZE + 100  # Extra space for controls
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("A* Pathfinding Visualizer")
        
        # Font for text
        self.font = pygame.font.Font(None, 24)
        
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
                # Set wall drawing mode based on current cell state
                if self.wall_drawing_mode is None:
                    self.wall_drawing_mode = 1 - self.grid[row][col]
                self.grid[row][col] = self.wall_drawing_mode
            elif self.mode == 'start':
                # Clear previous start
                if self.start:
                    old_row, old_col = self.start
                    if self.grid[old_row][old_col] == 0:
                        pass  # Keep as free space
                self.start = (row, col)
                self.grid[row][col] = 0  # Ensure start is free
            elif self.mode == 'goal':
                # Clear previous goal
                if self.goal:
                    old_row, old_col = self.goal
                    if self.grid[old_row][old_col] == 0:
                        pass  # Keep as free space
                self.goal = (row, col)
                self.grid[row][col] = 0  # Ensure goal is free
                
    def handle_mouse_motion(self, pos):
        if self.mouse_pressed and self.mode == 'wall':
            cell = self.get_cell_from_pos(pos)
            if cell:
                row, col = cell
                if self.wall_drawing_mode is not None:
                    self.grid[row][col] = self.wall_drawing_mode
                
    def find_path(self):
        if self.start and self.goal:
            self.path = a_star_search(self.start, self.goal, self.grid) or []
            
    def draw_grid(self):
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
                elif (row, col) in self.path:
                    color = COLORS['yellow'] # Path
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
            self.screen.blit(text_surface, (10, y_offset + i * 25))
            
        # Instructions
        instructions = [
            "SPACE - Find Path",
            "R - Reset Grid",
            "C - Clear Path"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = self.font.render(text, True, COLORS['black'])
            self.screen.blit(text_surface, (200, y_offset + i * 25))
            
    def reset_grid(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.start = None
        self.goal = None
        self.path = []
        
    def clear_path(self):
        self.path = []
        
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
                    elif event.key == pygame.K_SPACE:
                        self.find_path()
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
    visualizer = AStarVisualizer(12, 12)
    visualizer.run()