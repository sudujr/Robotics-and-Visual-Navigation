import pygame

class Display:
    """
        Class based implementation for displaying output using pygame as videos are series of images 
    """
    def __init__(self, H, W):
        pygame.init()
        self.white = (255, 255, 255)
        self.screen = pygame.display.set_mode((W,H))

    def view(self, frame):
        surf = pygame.surfarray.make_surface(frame.swapaxes(0,1)).convert()
        self.screen.fill(self.white)
        self.screen.blit(surf,(0,0))
        pygame.display.update()