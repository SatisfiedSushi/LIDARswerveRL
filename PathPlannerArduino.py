import pygame
import sys
import json
import numpy as np

width = 1000  # Original width of the window
height = 1000  # Original height of the window
# Scale factor for the application adjusts based on the original window size
scale = width / 200

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((width, height))  # Scale the window size
pygame.display.set_caption("Draw a Path")
clock = pygame.time.Clock()

# Variables
running = True
drawing = False  # True when the mouse is held down
points = []  # Store points of the path
start_point = None  # Store the start point
end_point = None  # Store the end point

# Grid size
grid_size = 4

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Stop drawing when Enter is pressed
                running = False
            elif event.key == pygame.K_s:  # Set start point when 'S' is pressed
                start_point = pygame.mouse.get_pos()
            elif event.key == pygame.K_e:  # Set end point when 'E' is pressed
                end_point = pygame.mouse.get_pos()
            elif event.key == pygame.K_c:  # Clear grid when 'C' is pressed
                points = []
                start_point = None
                end_point = None

    if drawing:
        # Get the current mouse position
        position = pygame.mouse.get_pos()
        if not points or points[-1] != position:
            points.append((position[0], position[1]))

    # Clear screen
    screen.fill((255, 255, 255))

    # Draw the points
    if len(points) > 1:
        pygame.draw.lines(screen, (0, 0, 0), False, [(x, y) for x, y in points], 2)

    # Draw the start and end points
    if start_point:
        pygame.draw.circle(screen, (0, 255, 0), (start_point[0], start_point[1]), 5)
    if end_point:
        pygame.draw.circle(screen, (255, 0, 0), (end_point[0], end_point[1]), 5)

    # Draw the grid
    for i in range(grid_size):
        # Draw horizontal line
        pygame.draw.line(screen, (0, 0, 0), (0, i * height / grid_size), (width, i * height / grid_size), 10)
        # Draw vertical line
        pygame.draw.line(screen, (0, 0, 0), (i * width / grid_size, 0), (i * width / grid_size, height), 10)

    pygame.display.flip()
    clock.tick(60)

# Scale the points back to the original window size
points = [(x/scale, y/scale) for x, y in points]

# Save points to a JSON file
with open('path_points.json', 'w') as f:
    json.dump(points, f)

pygame.quit()

#Process the path

# Parameters (example values, adjust according to your setup)
wheel_diameter_cm = 10.0
gear_ratio = 1.0
encoder_ticks_per_revolution = 360

def calculate_motor_speeds(point_a, point_b):
    # Calculate the distance between two points
    distance = np.linalg.norm(np.array(point_b) - np.array(point_a))
    # Simplified speed calculation (adjust as needed)
    speed = min(distance / 10.0, 1.0)  # Normalize speed (0-1)
    return speed, speed  # Assuming straight movement for simplicity

# Calculate motor speeds for each segment
path_with_speeds = []
for i in range(1, len(points)):
    speed_left, speed_right = calculate_motor_speeds(points[i - 1], points[i])
    path_with_speeds.append({
        "speed_left": speed_left,
        "speed_right": speed_right
    })

# Save the path data to a JSON file
with open('path_data.json', 'w') as f:
    json.dump(path_with_speeds, f, indent=4)
