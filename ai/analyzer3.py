import os
import numpy as np
import matplotlib.pyplot as plt
import gc


from ai_player import AI_Player


class Analyzer3(object):
    def __init__(self, data_path, section_id, img_diversity, wrd_diversity, content):
        self.data_path = data_path
        self.section_id = section_id
        self.img_diversity = img_diversity
        self.wrd_diversity = wrd_diversity
        self.player = AI_Player(0)
        self.wrd_vec_data = self.load_wrd_vec()
        if content == "wrd_diversity":
            self.plot_affect_of_wrd_diversity(self.img_diversity, self.wrd_diversity)
        elif content == "img_diversity":
            self.plot_affect_of_img_diversity(self.img_diversity, self.wrd_diversity)
        
    
    def load_data(self, img_diversity, wrd_diversity):
        data = os.path.join(self.data_path, f"{img_diversity}-{wrd_diversity}.npy")
        data = np.load(data, allow_pickle=True)

        return data

    def plot_affect_of_wrd_diversity(self, img_diversity, wrd_diversity):
        data = self.load_data(img_diversity, wrd_diversity)
        section_data = data[self.section_id]
        x, y = [], [] # actual_sim, sim_from_wrd
        for i, d in enumerate(section_data):
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
        print("lll")
        print(res)
        
    def plot_affect_of_img_diversity(self, img_diversity, wrd_diversity):
        data = self.load_data(img_diversity, wrd_diversity)
        section_data = data[self.section_id]
        x, y = [], [] # actual_sim, sim_from_img
        for i, d in enumerate(section_data):
            if i == 0:
                pre_wrd = d
            elif i % 2 == 1:
                guess_wrd = self.player.guess(d, 0, 0)
                sim_from_img = self.calc_cos_sim(pre_wrd, guess_wrd)
                y.append(sim_from_img)
            else:
                actual_sim = self.calc_cos_sim(pre_wrd, d)
                pre_wrd = d
                x.append(actual_sim)
        
        res = [x, y]

        print(res)

    def calc_cos_sim(self, w1, w2):
        v1 = self.wrd_vec_data[w1]
        v2 = self.wrd_vec_data[w2]

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def load_wrd_vec(self):
        load_wrd_vec = np.load("word_vector.npz")
        wrd_vec_list = load_wrd_vec["wrd_vec_list"]
        wrd_vec_data = {}
        with open("imagenet_classes.txt") as f:
            for i, line in enumerate(f.readlines()):
                wrd = line.replace("\n", "")
                wrd_vec_data[wrd] = wrd_vec_list[i]

        return wrd_vec_data

if __name__ == "__main__":
    data_path = str(input())
    section_id = int(input())
    img_diversity = str(input())
    wrd_diversity = str(input())
    content = str(input())
    analyzer = Analyzer3(data_path, section_id, img_diversity, wrd_diversity, content)


