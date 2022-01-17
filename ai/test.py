from ai_player import AI_Player

import numpy as np

class Test(object):
    def __init__(self, step_n, round_n):
        self.step_n = step_n
        self.round_n = round_n
        self.player = AI_Player(0)
        self.wrd_books = []

        for i in range(self.round_n):
            print("round "+str(i+1)+"........")
            self.round()

    def round(self):

        self.sketch_book = []
        self.wrd_book = []
        self.make_secret_wrd()

        for i in range(self.step_n):

            # sketch turn
            if i % 2 == 0:
                sketch = self.player.sketch(self.sketch_book[i], i, truncation=1)
                self.sketch_book.append(sketch)

            else:
                guess = self.player.guess(self.sketch_book[i], i)
                self.sketch_book.append(guess)
                self.wrd_book.append(guess)

        self.wrd_books.append(self.wrd_book)
    
    def make_secret_wrd(self):
        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        wrd_idx = np.random.randint(0, 1000)
        self.sketch_book.append(wrds[wrd_idx])
        self.wrd_book.append(wrds[wrd_idx])

if __name__ == "__main__":
    test = Test(10, 2)
    print(test.wrd_books)