import itertools
import numpy as np
from tqdm import tqdm

from ai_player import AI_Player

class Experiment(object):
    def __init__(self, truncation, noise_degree, noise_rate, step_num, iteration, iter_count):
        self.truncation = truncation
        self.noise_degree = noise_degree
        self.noise_rate = noise_rate
        self.step_num = step_num
        self.iteration = iteration
        self.iter_count = iter_count

        self.player = AI_Player(0)

    def make_secret_wrd(self):
        with open("imagenet_classes.txt") as f:
            wrds = [line.strip() for line in f.readlines()]
        
        secret_wrd_idx = np.random.randint(0, 1000)
        while secret_wrd_idx in self.player.unavailable_wrd_ids:
            secret_wrd_idx = np.random.randint(0, 1000)
        
        return wrds[secret_wrd_idx]

    def conduct(self):
        res = []
        for iter in range(self.iteration):
            section_res = []
            section_res.append(self.make_secret_wrd())
            for step in range(self.step_num):
                if step % 2 == 0:
                    img_path = self.player.sketch(section_res[step], step, self.truncation, self.noise_degree, iter+self.iter_count)
                    section_res.append(img_path)
                else:
                    guessed = self.player.guess(section_res[step], self.noise_rate, self.noise_degree)
                    section_res.append(guessed)
            
            res.append(section_res)
        print(res)
        return res

if __name__ == "__main__":
    truncation = float(input("truncation: "))
    noise_degree = float(input("noise degree: "))
    noise_rate = float(input("noise rate: "))
    iteration = int(input("iteration: "))
    iter_count = int(input("iter count: "))
    step_num = 50

    experiment = Experiment(truncation, noise_degree, noise_rate, step_num, iteration, iter_count)
    experiment.conduct()