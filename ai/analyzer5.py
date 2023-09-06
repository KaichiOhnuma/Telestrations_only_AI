import os
import numpy as np

class Analyzer5(object):
    def __init__(self, data_path, img_diversity, wrd_diversity):
        self.data_path = data_path
        self.img_diversity = img_diversity
        self.wrd_diversity = wrd_diversity
        self.wrd_vec_data = self.load_wrd_vec()
        self.data = self.load_data()

        self.pick_up_five_biggest_miss()

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
        
        return data

    def pick_up_five_biggest_miss(self):
        all_sim_list = np.array([])
        for section_d in self.data:
            for i, d in enumerate(section_d):
                if i == 0:
                    pre_wrd = d
                elif i % 2 == 0:
                    sim = self.calc_cos_sim(pre_wrd, d)
                    sim_list = np.append(section_d[i-2:i+1], sim)
                    all_sim_list = np.append(all_sim_list, sim_list)
                    pre_wrd = d

        all_sim_list = all_sim_list.reshape(1250, 4)
        all_sim_list = all_sim_list[np.argsort(all_sim_list[:, 3])]
        print(all_sim_list[:5])
                        
            
    def calc_cos_sim(self, w1, w2):
        v1 = self.wrd_vec_data[w1]
        v2 = self.wrd_vec_data[w2]

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    data_path = "../data/20220924"
    img_diversity = 2.5
    wrd_diversity = 1.0
    analyzer = Analyzer5(data_path, img_diversity, wrd_diversity)

    print("--------------------------------------------------------------------------------------------------")

    img_diversity = 1.0
    wrd_diversity = 1.5
    analyzer = Analyzer5(data_path, img_diversity, wrd_diversity)

