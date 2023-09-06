import os
import numpy as np

from ai_player import AI_Player

class Analyzer4(object):
    def __init__(self, data_path, section_id, img_diversity, wrd_diversity, category):
        self.data_path = data_path
        self.section_id = section_id
        self.img_diversity = img_diversity
        self.wrd_diversity = wrd_diversity
        self.category = category
        self.wrd_vec_data = self.load_wrd_vec()
        self.data = self.load_data()
        self.player = AI_Player(0)

        if category == "save_img_dei_scatter":
            self.save_img_dei_scatter()
        elif category == "save_wrd_dei_scatter":
            self.save_wrd_dei_scatter()
        elif category == "get_avg":
            self.get_avg()
        elif category == "get_variance":
            self.get_variance()
        else:
            print("error")

    def load_wrd_vec(self):
        load_wrd_vec = np.load("word_vector.npz")
        wrd_vec_list = load_wrd_vec["wrd_vec_list"]
        wrd_vec_data = {}
        with open("imagenet_classes.txt") as f:
            for i, line in enumerate(f.readlines()):
                wrd = line.replace("\n", "")
                wrd_vec_data[wrd] = wrd_vec_list[i]

        return wrd_vec_data

    def load_data(self):
        data = os.path.join(self.data_path, f"{self.img_diversity}-{self.wrd_diversity}.npy")
        data = np.load(data, allow_pickle=True)
        data = data[section_id]
        
        return data

    def save_img_dei_scatter(self):
        x, y = [], [] # actual_sim, sim_from_img
        for i, d in enumerate(self.data):
            if i == 0:
                pre_wrd = d
            elif i % 2 == 1:
                guess_wrd = self.player.guess(d, 0, 0)
                sim_from_wrd = self.calc_cos_sim(pre_wrd, guess_wrd)
                y.append(sim_from_wrd)
            else:
                actual_sim = self.calc_cos_sim(pre_wrd, d)
                x.append(actual_sim)

        res = [x, y]
        print(res)

    def save_wrd_dei_scatter(self):
        x, y = [], [] # actual_sim, sim_from_wrd
        for i, d in enumerate(self.data):
            if i == 0:
                pre_wrd = d
            elif i % 2 == 1:
                guess_wrd = self.player.guess(d, 0, 0)
            else:
                actual_sim = self.calc_cos_sim(pre_wrd, d)
                sim_from_wrd = self.calc_cos_sim(guess_wrd, d)
                x.append(actual_sim)
                y.append(sim_from_wrd)
        
        res = [x, y]
        print(res)

    def calc_cos_sim(self, w1, w2):
        v1 = self.wrd_vec_data[w1]
        v2 = self.wrd_vec_data[w2]

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_avg(self):
        avg_actual_sim, avg_sim_from_img, avg_sim_from_wrd = 0, 0, 0
        for i, d in enumerate(self.data):
            if i == 0:
                pre_wrd = d
            elif i % 2 == 1:
                guess_wrd = self.player.guess(d, 0, 0)
                avg_sim_from_img += self.calc_cos_sim(pre_wrd, guess_wrd)
            else:
                avg_actual_sim += self.calc_cos_sim(pre_wrd, d)
                avg_sim_from_wrd += self.calc_cos_sim(guess_wrd, d)
        avg_actual_sim /= (len(self.data) - 1 ) / 2
        avg_sim_from_img /= (len(self.data) - 1 ) / 2
        avg_sim_from_wrd /= (len(self.data) - 1 ) / 2
        res = [avg_actual_sim, avg_sim_from_img, avg_sim_from_wrd]
        print(res)

if __name__ == "__main__":
    data_path = str(input())
    section_id = int(input())
    img_diversity = str(input())
    wrd_diversity = str(input())
    category = str(input())
    analyzer = Analyzer4(data_path, section_id, img_diversity, wrd_diversity, category)
