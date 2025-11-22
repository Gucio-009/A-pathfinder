# ----------------------------------------INSTRUCTION------------------------------------
# LEFT BUTTON MOUSE - first mark start point, secound end point and than you paint barriers
# RIGHT BUTTON MOUSE - you can unmark point to reset
# SPACE - press space after you marked star, end and barriers to start the algorithm


import pygame
import math
from queue import PriorityQueue



# Window setup
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('A* pathfinding')

# Colors used in visualization
RED = (255, 0, 0)           # closed nodes
GREEN = (0, 255, 0)         # open nodes
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)           # barriers
PURPLE = (128, 0, 128)      # final path
ORANGE = (255, 165, 0)      # start node
TURQUOISE = (64, 224, 208)  # end node
GREY = (128, 128, 128)

# Each square on the grid
class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width      # pixel position X
        self.y = col * width      # pixel position Y
        self.color = WHITE
        self.neighbors = []       # accessible neighbors
        self.width = width
        self.total_rows = total_rows

    # Returns (row, col) for heuristic
    def get_pos(self):
        return self.row, self.col

    # State checks
    def is_closed(self): return self.color == RED
    def is_open(self): return self.color == GREEN
    def is_barrier(self): return self.color == BLACK
    def is_start(self): return self.color == ORANGE
    def is_end(self): return self.color == TURQUOISE

    # Reset node color
    def reset(self): self.color = WHITE

    # Marking methods for visualization
    def make_closed(self): self.color = RED
    def make_open(self): self.color = GREEN
    def make_barrier(self): self.color = BLACK
    def make_end(self): self.color = TURQUOISE
    def make_start(self): self.color = ORANGE
    def make_path(self): self.color = PURPLE

    # Draws this square
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    # Finds all accessible neighbors
    def update_neighbors(self, grid):
        self.neighbors = []

        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        # Needed for PriorityQueue but unused
        return False

# Heuristic function (Manhattan distance)
def h(n1, n2):
    x1, y1 = n1
    x2, y2 = n2
    return abs(x1 - x2) + abs(y1 - y2)

# Builds the final path by moving backwards from end
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

# A* algorithm implementation
def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()        # nodes to be evaluated
    open_set.put((0, count, start))   # push start to queue

    came_from = {}                    # keeps track of best parents

    # Initialize g_score and f_score for all nodes
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0

    f_score = {node: float('inf') for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    # Auxiliary set to check if node is already in open_set
    open_set_hash = {start}

    while not open_set.empty():
        # Allows quitting during algorithm execution
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Extract node with lowest f_score
        current = open_set.get()[2]
        open_set_hash.remove(current)

        # Goal reached
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        # Check neighbors
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1   # movement cost

            # Found a better path
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

                # Add neighbor to open_set if not already there
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        # Node fully evaluated
        if current != start:
            current.make_closed()

    return False

# Creates all nodes of the grid
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

# Draws the grid lines
def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

# Draws the entire screen
def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

# Converts mouse click to grid position
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

# Main loop
def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # LEFT CLICK → place start, end, or barrier
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]

                if not start and node != end:
                    start = node
                    start.make_start()

                elif not end and node != start:
                    end = node
                    end.make_end()

                elif node != end and node != start:
                    node.make_barrier()

            # RIGHT CLICK → reset node
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()

                if node == start:
                    start = None
                elif node == end:
                    end = None

            # Keys
            if event.type == pygame.KEYDOWN:
                # SPACE → run A*
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                # C → clear board
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

# Run program
main(WIN, WIDTH)


