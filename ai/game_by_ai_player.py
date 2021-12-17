"""
Handles operations related to game by AI players
"""

from ai_player import AI_Player
import numpy as np

class Game_by_AI_player(object):
    def __init__(self, player_num):
        """
        init the game
        :param player_num: int
        """
        self.player_num = player_num
        self.round_count = 0
        self.players = [AI_Player(i) for i in range(self.player_num)]
        self.sketch_books = [[] for _ in range(self.player_num)]
        self.make_secret_word()
        self.play_game()
    
    def play_game(self):
        """
        play one game
        :return: None
        """

        while self.round_count < self.player_num:
            self.round_count += 1
            self.round()

        self.round_count = 0
            
        
    def make_secret_word(self):
        """
        make a secret word for each player
        :return: None
        """
        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        for i in range(self.player_num):
            wrd_idx = np.random.randint(0, 1000)
            self.sketch_books[i].append(wrds[wrd_idx])

    def round(self):
        """
        start a new round
        :return: None
        """

        for i in range(self.player_num):
            sketch_book_idx = (i + self.round_count -1) % self.player_num

            # sketch turn
            if self.round_count % 2 == 1:
                sketch = self.players[i].sketch(self.sketch_books[sketch_book_idx][self.round_count-1], self.round_count)
                self.sketch_books[sketch_book_idx].append(sketch)
            
            # guess turn
            else:
                guess = self.players[i].guess(self.sketch_books[sketch_book_idx][self.round_count-1], self.round_count)
                self.sketch_books[sketch_book_idx].append(guess)

    def draw(self):
        """
        draw sketchbooks when the game is done
        :return: None
        """
        import pygame

        self.BG = (255,255,255)
        self.WIDTH = 1000
        self.HEIGHT = 600
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT+30))
        pygame.font.init()
        self.guess_font = pygame.font.SysFont("timesnewroman", 20, bold=True)
        self.arrow_font = pygame.font.SysFont("timesnewroman", 50, bold=True)
        self.win.fill(self.BG)

        for i in range(self.player_num):

            for j in range(self.player_num+1):

                if j % 2 == 0:
                    guess = self.sketch_books[i][j].split(',')[0]
                    guess = self.guess_font.render(guess, 1, (0,0,0))
                    self.win.blit(guess, (j*self.WIDTH/(self.player_num+1) - guess.get_width()/2 + self.WIDTH/(self.player_num+1)/2, i*self.HEIGHT/self.player_num - guess.get_height()/2 + self.HEIGHT/self.player_num/2 + i*10))
                else:
                    img = pygame.image.load(self.sketch_books[i][j])
                    img = pygame.transform.scale(img, (self.WIDTH/(self.player_num+1), self.HEIGHT/self.player_num))
                    self.win.blit(img, (j*self.WIDTH/(self.player_num+1) - img.get_width()/2 + self.WIDTH/(self.player_num+1)/2, i*self.HEIGHT/self.player_num - img.get_height()/2 + self.HEIGHT/self.player_num/2 + i*10))

                if j != 0:
                    arrow = self.arrow_font.render(">", 1, (0,0,0))
                    self.win.blit(arrow, (j*self.WIDTH/(self.player_num+1) - arrow.get_width()/2, i*self.HEIGHT/self.player_num - arrow.get_height()/2 + self.HEIGHT/self.player_num/2 + i*10))

        pygame.display.update()
        

if __name__ == "__main__":
    game = Game_by_AI_player(4)
    while True:
        game.draw()