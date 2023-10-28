import matplotlib.pyplot as plt
import math
from heapq import heappop, heappush
import time

def create_grid(m, n, obstacles, start, goal):
    grid = [['.' for _ in range(n)] for _ in range(m)]

    for obstacle in obstacles:
        x, y = obstacle
        grid[x][y] = 'X'

    # Ensure the start point is within the grid boundaries
    start = (max(0, min(start[0], m - 1)), max(0, min(start[1], n - 1)))

    grid[start[0]][start[1]] = 'S'
    grid[goal[0]][goal[1]] = 'G'

    return grid

def plot_grid_with_path(grid, path=None):
    rows = len(grid)
    cols = len(grid[0])

    plt.figure(figsize=(cols, rows))

    # Define numerical values for different cell types
    cell_values = {'X': 0, '.': 1, 'S': 2, 'G': 3}

    # Use a colormap that differentiates between cell types
    cmap = plt.cm.colors.ListedColormap(['black', 'purple', 'purple', 'purple'])

    # Convert grid to numerical values based on cell type
    num_grid = [[cell_values[cell] for cell in row] for row in grid]

    plt.imshow(num_grid, cmap=cmap, vmin=0, vmax=3)  # Set vmin and vmax for color mapping

    # Draw grid lines around each cell
    for i in range(rows + 1):
        plt.axhline(y=i - 0.5, color='white', linewidth=0.5)
    for j in range(cols + 1):
        plt.axvline(x=j - 0.5, color='white', linewidth=0.5)

    # Define custom x and y axis labels based on grid values
    x_labels = [str(j) for j in range(cols)]
    y_labels = [str(i) for i in range(rows)]

    plt.xticks(range(cols), x_labels)
    plt.yticks(range(rows), y_labels)

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'S':
                plt.text(j, i, 'S', color='white', ha='center', va='center')
            elif grid[i][j] == 'G':
                plt.text(j, i, 'G', color='white', ha='center', va='center')

    if path:
        path_x = [point[1] for point in path]
        path_y = [point[0] for point in path]
        plt.plot(path_x, path_y, marker='o', markersize=8, color='red')

    plt.show()

def a_star(grid, start, goal, heuristic_func):
    open_list = [(0, start)]
    came_from = {}
    g_score = {point: float('inf') for point in [(i, j) for i in range(len(grid)) for j in range(len(grid[0]))]}
    g_score[start] = 0
    f_score = {point: float('inf') for point in [(i, j) for i in range(len(grid)) for j in range(len(grid[0]))]}
    f_score[start] = heuristic_func(start, goal)

    while open_list:
        _, current = heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            path.insert(0, start)
            return path

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][
                neighbor[1]] != 'X':
                cost = 1

                tentative_g_score = g_score[current] + cost
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_func(neighbor, goal)
                    heappush(open_list, (f_score[neighbor], neighbor))

    return None

# Heuristic functions
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def diagonal_distance(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def print_grid(grid):
    for row in grid:
        print(' '.join(row))

# Get input from the console
def get_console_input():
    # Get input for grid size (m and n)
    m, n = map(int, input("Enter grid size (m n): ").split())

    # Get the number of obstacles (k)
    k = int(input("Enter the number of obstacles (k): "))

    # Initialize an empty list to store obstacle coordinates
    obstacles = []

    # Get obstacle coordinates (x, y) for k obstacles
    print("Enter obstacle coordinates (x y) for each obstacle:")
    for _ in range(k):
        x, y = map(int, input().split())
        obstacles.append((x, y))

    # Get start and goal coordinates
    start = tuple(map(int, input("Enter the start coordinates (startx starty): ").split()))
    goal = tuple(map(int, input("Enter the goal coordinates (goalx goaly): ").split()))

    return m, n, obstacles, start, goal

# Get input from the console
m, n, obstacles, start, goal = get_console_input()

# Create and print the grid with start and goal points
grid = create_grid(m, n, obstacles, start, goal)
print("Grid:")
print_grid(grid)

# Heuristic functions (same as before)

heuristics = {
    "Manhattan Distance": manhattan_distance,
    "Diagonal Distance": diagonal_distance,
    "Euclidean Distance": euclidean_distance
}

# Apply A* algorithm with different heuristics and plot the paths

for name, heuristic_func in heuristics.items():
    # Start measuring time
    start_time = time.perf_counter()

    path = a_star(grid, start, goal, heuristic_func)

    # Stop measuring time
    end_time = time.perf_counter()

    if path:
        print(f"\nPath using {name}:")
        total_cost = len(path) - 1  # Total path cost is the number of cells traversed
        print("Total Path Cost (in cells):", total_cost)

        # Include the starting node 'S' in the cell path
        cell_path = [start] + path[1:]  # Include start at the beginning
        print("Cell Path:", cell_path)

        runtime_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        print("Runtime:", "{:.6f}".format(runtime_ms), "milliseconds")
        plot_grid_with_path(grid, path)
    else:
        print(f"\nNo path found using {name}")
        