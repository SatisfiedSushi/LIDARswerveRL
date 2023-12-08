import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
SPEED = 0.2
ROTATE_SPEED = 0.1

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Swerve Drive Example")

# Initial position and angle
x, y = WIDTH // 2, HEIGHT // 2
angle = 0

def rotate_point(point, angle, pivot):
    """Rotate a point around a pivot by a given angle in degrees."""
    radian_angle = math.radians(angle)
    x, y = point
    pivot_x, pivot_y = pivot
    new_x = pivot_x + (x - pivot_x) * math.cos(radian_angle) - (y - pivot_y) * math.sin(radian_angle)
    new_y = pivot_y + (x - pivot_x) * math.sin(radian_angle) + (y - pivot_y) * math.cos(radian_angle)
    return new_x, new_y

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Rotation based on left and right arrow keys
    if keys[pygame.K_LEFT]:
        angle += ROTATE_SPEED
    if keys[pygame.K_RIGHT]:
        angle -= ROTATE_SPEED

    # Movement based on WASD keys
    if keys[pygame.K_a]:
        new_x = x + SPEED * math.cos(math.radians(angle))
        new_y = y - SPEED * math.sin(math.radians(angle))
        if 0 <= new_x <= WIDTH - 50 and 0 <= new_y <= HEIGHT - 50:
            x, y = new_x, new_y

    if keys[pygame.K_d]:
        new_x = x - SPEED * math.cos(math.radians(angle))
        new_y = y + SPEED * math.sin(math.radians(angle))
        if 0 <= new_x <= WIDTH - 50 and 0 <= new_y <= HEIGHT - 50:
            x, y = new_x, new_y

    if keys[pygame.K_w]:
        new_x = x - SPEED * math.sin(math.radians(angle))
        new_y = y - SPEED * math.cos(math.radians(angle))
        if 0 <= new_x <= WIDTH - 50 and 0 <= new_y <= HEIGHT - 50:
            x, y = new_x, new_y

    if keys[pygame.K_s]:
        new_x = x + SPEED * math.sin(math.radians(angle))
        new_y = y + SPEED * math.cos(math.radians(angle))
        if 0 <= new_x <= WIDTH - 50 and 0 <= new_y <= HEIGHT - 50:
            x, y = new_x, new_y

    screen.fill((255, 255, 255))
    rotated_box = pygame.transform.rotate(pygame.Surface((50, 50)), -angle)
    screen.blit(rotated_box, (x, y))

    pygame.display.flip()

pygame.quit()
sys.exit()
