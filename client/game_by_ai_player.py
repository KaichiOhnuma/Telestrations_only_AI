"""
Handles operations related to game by AI players
"""

from ai_player import AI_Player

class Game_by_AI_player(object):
    def __init__(self, player_num):
        self.player_num = player_num
        self.players = [AI_Player(i) for i in range(self.player_num)]
        self.sketch_books = [[] for _ in range(self.player_num)]
        self.make_secret_word()
        self.round_count = 0

        while self.round_count < self.player_num:
            self.round_count += 1
            self.round()
            
        
    def make_secret_word(self):
        self.sketch_books[0].append("punching bag")
        self.sketch_books[1].append("fruit cake")
        self.sketch_books[2].append("wizard")
        self.sketch_books[3].append("aquarium")

    def round(self):

        for i in range(self.player_num):
            sketch_book_index = (i + self.round_count -1) % self.player_num
            if self.round_count % 2 == 1:
                sketch = self.players[i].sketch(self.sketch_books[sketch_book_index][self.round_count-1])
                self.sketch_books[sketch_book_index].append(sketch)
            else:
                guess = self.players[i].guess(self.sketch_books[sketch_book_index][self.round_count-1])
                self.sketch_books[sketch_book_index].append(guess)

    def get_sketch_books(self):
        return self.sketch_books

test = Game_by_AI_player(4)
sketch_books = test.get_sketch_books()
print(sketch_books)