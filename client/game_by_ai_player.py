"""
Handles operations related to game by AI players
"""

from ai_player import AI_Player
import pygame

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

        while self.round_count < self.player_num:
            self.round_count += 1
            self.round()
            
        
    def make_secret_word(self):
        """
        make a secret word for each player
        :return: None
        """
        self.sketch_books[0].append("punching bag")
        self.sketch_books[1].append("fruit cake")
        self.sketch_books[2].append("wizard")
        self.sketch_books[3].append("aquarium")

    def round(self):
        """
        start a new round
        :return: None
        """

        for i in range(self.player_num):
            sketch_book_index = (i + self.round_count -1) % self.player_num
            if self.round_count % 2 == 1:
                sketch = self.players[i].sketch(self.sketch_books[sketch_book_index][self.round_count-1], self.round_count)
                self.sketch_books[sketch_book_index].append(sketch)
            else:
                guess = self.players[i].guess(self.sketch_books[sketch_book_index][self.round_count-1])
                self.sketch_books[sketch_book_index].append(guess)

    def draw(self):
        """
        draw sketchbooks when the game is done
        :return: None
        """
        self.BG = (255,255,255)
        self.WIDTH = 1000
        self.HEIGHT = 600
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.font.init()
        self.guess_font = pygame.font.SysFont("Times New Roman", 30, bold=True)
        self.arrow_font = pygame.font.SysFont("Times New Roman", 50, bold=True)
        self.win.fill(self.BG)

        for i in range(self.player_num):

            for j in range(self.player_num+1):

                if j % 2 == 0:
                    guess = self.guess_font.render(self.sketch_books[i][j], 1, (0,0,0))
                    self.win.blit(guess, (j*self.WIDTH/(self.player_num+1) - guess.get_width()/2 + self.WIDTH/(self.player_num+1)/2, i*self.HEIGHT/self.player_num - guess.get_height()/2 + self.HEIGHT/self.player_num/2))
                else:
                    img = pygame.image.load(self.sketch_books[i][j])
                    img = pygame.transform.scale(img, (self.WIDTH/(self.player_num+1), self.HEIGHT/self.player_num))
                    self.win.blit(img, (j*self.WIDTH/(self.player_num+1) - img.get_width()/2 + self.WIDTH/(self.player_num+1)/2, i*self.HEIGHT/self.player_num - img.get_height()/2 + self.HEIGHT/self.player_num/2))

                if j != 0:
                    arrow = self.arrow_font.render("->", 1, (0,0,0))
                    self.win.blit(arrow, (j*self.WIDTH/(self.player_num+1) - arrow.get_width()/2, i*self.HEIGHT/self.player_num - arrow.get_height()/2 + self.HEIGHT/self.player_num/2))

        pygame.display.update()
        

if __name__ == "__main__":
    game = Game_by_AI_player(4)
    while True:
        game.draw()