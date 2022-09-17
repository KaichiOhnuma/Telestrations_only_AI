"""
Represent object of Evaluation test for AI player
"""

from ai_player import AI_Player

import numpy as np
import sys
import tkinter as tk
from PIL import Image, ImageTk

class Test(object):
    def __init__(self):
        """
        init the Evaluation test object
        :param step_n: int 
        :param iteration: int 
        """
        self.step_n = int(input("step_n: "))
        self.iteration = int(input("iteration: "))
        self.truncation = float(input("truncation: "))
        self.noise_degree = float(input("noise degree: "))
        self.noise_rate = float(input("noise rate: "))

        self.player = AI_Player(0)
        self.passed_wrd_list = []

        #wrd_vec_file = np.load("word_vector.npz")
        #self.unavailable_wrd_idxs = wrd_vec_file["unavailable_wrd_idxs"]
        #self.wrd_vec_list = wrd_vec_file["wrd_vec_list"]
        self.main()
        #self.display_test(0.75, 3, 1, 32)

    def round(self):
        """
        do one round
        :return: None
        """
        self.sketch_book = []
        self.passed_wrd = []
        self.make_secret_wrd()

        for i in range(self.step_n):

            # sketch turn
            if i % 2 == 0:
                sketch = self.player.sketch(self.sketch_book[i], round_count=i+1, truncation=self.truncation)
                self.sketch_book.append(sketch)

            # guess turn
            else:
                guess = self.player.guess(self.sketch_book[i], noise_rate=self.noise_rate, noise_degree=self.noise_degree)
                self.sketch_book.append(guess)
                self.passed_wrd.append(guess)

        self.passed_wrd_list.append(self.passed_wrd)
    
    def make_secret_wrd(self):
        """
        make secret word
        """
        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        secret_wrd_idx = np.random.randint(0, 1000)
        while secret_wrd_idx in self.player.unavailable_wrd_ids:
            secret_wrd_idx = np.random.randint(0, 1000)

        self.sketch_book.append(wrds[secret_wrd_idx])
        self.passed_wrd.append(wrds[secret_wrd_idx])

    def evaluate_process(self):

        self.sim_to_previous_wrd = []
        self.sim_to_secret_wrd = []

        self.avg_number_of_wrd = 0
        self.avg_sim_to_p_wrd = 0
        self.avg_sim_to_s_wrd = 0
        self.avg_final_sim_to_s_wrd = 0

        failed = 0
        num_of_wrd = []


        for passed_wrd in self.passed_wrd_list:
            secret_wrd = passed_wrd[0]
            secret_wrd_idx = self.player.wrds.index(secret_wrd)
            secret_wrd_vec = self.wrd_vec_list[secret_wrd_idx]
            previous_wrd_vec = secret_wrd_vec
            previous_wrd_idx = secret_wrd_idx

            passed_wrd_idxs = []
            sim_to_p_wrd = []
            sim_to_s_wrd = []
            
            for current_wrd in passed_wrd:
                current_wrd_idx = self.player.wrds.index(current_wrd)
                current_wrd_vec = self.wrd_vec_list[current_wrd_idx]
                sim_to_s_wrd.append(self.get_cos_sim(secret_wrd_vec, current_wrd_vec))
                sim_to_p_wrd.append(self.get_cos_sim(previous_wrd_vec, current_wrd_vec))
                
                if previous_wrd_idx != current_wrd_idx:
                    failed += 1
                if not current_wrd_idx in passed_wrd_idxs:
                    passed_wrd_idxs.append(current_wrd_idx)

                previous_wrd_idx = current_wrd_idx
                previous_wrd_vec = current_wrd_vec

            self.sim_to_previous_wrd.append(sim_to_p_wrd)
            self.sim_to_secret_wrd.append(sim_to_s_wrd)
            num_of_wrd.append(len(passed_wrd_idxs))
        
        self.prob_of_failure = failed / (len(self.passed_wrd_list[0])-1) / self.iteration
        self.avg_number_of_wrd = sum(num_of_wrd) / len(num_of_wrd)

        for i in range(self.iteration):
            self.avg_sim_to_p_wrd += (sum(self.sim_to_previous_wrd[i]) - 1) / (len(self.sim_to_previous_wrd[i]) - 1) / self.iteration
            self.avg_sim_to_s_wrd += (sum(self.sim_to_secret_wrd[i]) - 1) / (len(self.sim_to_secret_wrd[i]) - 1) / self.iteration
            self.avg_final_sim_to_s_wrd += self.sim_to_secret_wrd[i][len(self.sim_to_secret_wrd[i])-1] / self.iteration

    def evaluate_process2(self):
        sim_to_secret_word = np.zeros(len(self.passed_wrd_list[0]))
        for passed_wrd in self.passed_wrd_list:
            for i, wrd in enumerate(passed_wrd):
                sim_to_secret_word[i] += self.get_cos_sim(self.player.wrd_dict[wrd]["vector"], self.player.wrd_dict[passed_wrd[0]]["vector"])
        sim_to_secret_word = sim_to_secret_word / len(self.passed_wrd_list)

        return sim_to_secret_word

    def get_cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def main(self):

        for i in range(self.iteration):
            self.round()
            #print(f"round : {i} ...")

        sim_to_secret_word = self.evaluate_process2()
        sys.stdout.write(str(sim_to_secret_word))

    def display_test(self, truncation, mutation_degree, mutation_rate, step_n):
        sketch_book = []

        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        secret_wrd_idx = np.random.randint(0, 1000)
        while secret_wrd_idx in self.unavailable_wrd_idxs:
            secret_wrd_idx = np.random.randint(0, 1000)
        sketch_book.append(wrds[secret_wrd_idx])

        for i in range(step_n):
            # sketch turn
            if i % 2 == 0:
                sketch = self.player.sketch(sketch_book[i], round_count=i+1, truncation=truncation)
                sketch_book.append(sketch)

            # guess turn
            else:
                guess = self.player.guess(sketch_book[i], round_count=i+1, mutation_rate=mutation_rate, mutation_degree=mutation_degree)
                sketch_book.append(guess)

        print(sketch_book)
        self.draw(sketch_book)

    def draw(self, sketch_book):
        baseGround = tk.Tk()
        baseGround.geometry('1250x600')
        imgs = []

        for i in range(len(sketch_book)):
            if i != 0:
                arrow = tk.Label(text='→', font=("MSゴシック", '20', 'bold'))
                arrow_x = (i % 6) * 200 + 15
                arrow_y = (i // 6) * 150 + 60
                arrow.place(x=arrow_x, y=arrow_y)

        for i, data in enumerate(sketch_book):
            if i % 2 == 0:
                data_x = (i % 6) * 200 + 50
                data_y = (i // 6) * 150
                data = data.split(',')[0]
                #data = tk.Label(text=data, font=('MSゴシック', '15', 'bold'))
                #data.place(x=data_x, y=data_y)
                canvas = tk.Canvas(bg='white', width=150, height=150)
                canvas.place(x=data_x, y=data_y)
                canvas.create_text(75,75,text=data, anchor='center', font=('MSゴシック', '15', 'bold'))
            else:
                data_x = (i % 6) * 200 + 50
                data_y = (i // 6) * 150 
                img = Image.open(open(data, 'rb'))
                img.thumbnail((150, 150), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                imgs.append(img)
                canvas = tk.Canvas(bg='white', width=150, height=150)
                canvas.place(x=data_x, y=data_y)
                canvas.create_image(0,0,image=img, anchor = tk.NW)
            

        baseGround.mainloop()

if __name__ == "__main__":   
    test = Test()
