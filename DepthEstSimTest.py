import pygame

# Initialize Pygame
pygame.init()
screen_width = 800
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))

# Define 2D objects
objects = [
    pygame.Rect(100, 100, 50, 100),  # Wall 1
    pygame.Rect(300, 150, 80, 50),  # Wall 2
    pygame.Rect(500, 50, 30, 200)   # Wall 3
]

# Define 1D viewport
viewport_x = 400
viewport_height = 200

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Clear screen

    # Render 2D objects
    for obj in objects:
        pygame.draw.rect(screen, (255, 255, 255), obj)

    # Render 1D view
    for y in range(viewport_height):
        ray_x = viewport_x
        ray_y = screen_height // 2 - viewport_height // 2 + y

        for obj in objects:
            if obj.colliderect(pygame.Rect(ray_x, ray_y, 1, 1)):
                pygame.draw.line(screen, (255, 0, 0), (viewport_x, y), (viewport_x, y + 1))
                break

    pygame.display.flip()

pygame.quit()