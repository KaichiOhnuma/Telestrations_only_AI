import numpy as np

import sys
sys.path.append("../../")
from ai.ai_player import AI_Player

class Experiment_20231022(object):
    def __init__(self, img_diversity, wrd_diversity, step_num, iteration, iter_count, output_path):
        self.img_diversity = img_diversity
        self.wrd_diversity = wrd_diversity
        self.step_num = step_num
        self.iteration = iteration
        self.iter_count = iter_count
        self.output_path = output_path

        self.player = AI_Player(0)

        self.main()

    def main(self):
        res = []
        for iter in range(self.iteration):
            one_play_res = []
            one_play_res.append(self.make_start_wrd())

            for step in range(self.step_num):
                if step % 2 == 0:
                    img_path = self.player.sketch(one_play_res[step], self.img_diversity, self.wrd_diversity, self.iter_count+iter, step, self.output_path)
                    one_play_res.append(img_path)
                else:
                    guessed_wrd = self.player.guess(one_play_res[step], wrd_diversity)
                    one_play_res.append(guessed_wrd)
            
            res.append(one_play_res)
        
        print(res)


    def make_start_wrd(self):
        with open("../../ai/imagenet_classes.txt") as f:
            wrds = [line.strip() for line in f.readlines()]
        
        start_wrd_idx = np.random.randint(0, 1000)
        while start_wrd_idx in self.player.unavailable_wrd_idxs:
            start_wrd_idx = np.random.randint(0, 1000)

        return wrds[start_wrd_idx]

if __name__ == "__main__":
    img_diversity = float(input("img diversity: "))
    wrd_diversity = float(input("wrd diversity: "))
    step_num = int(input("step num: "))
    iteration = int(input("iteration: "))
    iter_count = int(input("iter count: "))
    output_path = str(input("output path: "))

    experiment = Experiment_20231022(img_diversity, wrd_diversity, step_num, iteration, iter_count, output_path)