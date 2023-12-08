class CoordConverter:
    def __init__(self):
        self.PPM = 100.0  # pixels per meter
        self.SCREEN_WIDTH = self.meters_to_pixels(16.46)
        self.SCREEN_HEIGHT = self.meters_to_pixels(8.23)

    def meters_to_pixels(self, meters):
        return int(meters * self.PPM)

    def pixels_to_meters(self, pixels):
        return pixels / self.PPM

    def pygame_to_box2d(self, pos_pygame):
        x, y = pos_pygame
        x_box2d = self.pixels_to_meters(x)
        y_box2d = self.pixels_to_meters(self.SCREEN_HEIGHT - y)  # Invert and convert to meters
        return x_box2d, y_box2d

    def box2d_to_pygame(self, pos_box2d):
        x, y = pos_box2d
        x_pygame = self.meters_to_pixels(x)
        y_pygame = self.SCREEN_HEIGHT - self.meters_to_pixels(y) # Invert and convert to pixels

        return x_pygame, y_pygame
