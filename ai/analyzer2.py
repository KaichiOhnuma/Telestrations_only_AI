import numpy as np
import itertools
import os
import matplotlib.pyplot as plt

class Analyzer2(object):
    def __init__(self, data_path, truncations, noise_degrees):
        self.wrd_vec_data = self.load_wrd_vec()
        self.data_path = data_path
        self.truncations = truncations
        self.noise_degrees = noise_degrees
        
    def load_wrd_vec(self):
        load_wrd_vec = np.load("word_vector.npz")
        wrd_vec_list = load_wrd_vec["wrd_vec_list"]
        wrd_vec_data = {}
        with open("imagenet_classes.txt") as f:
            for i, line in enumerate(f.readlines()):
                wrd = line.replace("\n", "")
                wrd_vec_data[wrd] = wrd_vec_list[i]

        return wrd_vec_data

    def load_data(self, truncation, noise_degree):
        data = os.path.join(self.data_path, f"{truncation}-{noise_degree}.npy")
        data = np.load(data, allow_pickle=True)
        
        return data

    def calc_cos_sim(self, w1, w2):
        v1 = self.wrd_vec_data[w1]
        v2 = self.wrd_vec_data[w2]

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def calc_avg_sim(self, data):
        section_num = len(data)
        step_num = len(data[0])
        avg_sim = 0

        for section_data in data:
            pre_wrd = section_data[0]
            for i in range(2, step_num, 2):
                sim = self.calc_cos_sim(pre_wrd, section_data[i])
                avg_sim += sim
                pre_wrd = section_data[i]

        avg_sim = avg_sim / section_num / (step_num // 2)

        return avg_sim

    def calc_final_sim(self, data):
        section_num = len(data)
        final_sim = 0

        for section_data in data:
            final_sim += self.calc_cos_sim(section_data[0], section_data[-1])

        final_sim /= section_num

        return final_sim

    def calc_min_sim(self, data):
        min_sim = 1

        for section_data in data:
            pre_wrd = section_data[0]
            for i in range(2, len(section_data), 2):
                sim = self.calc_cos_sim(pre_wrd, section_data[i])
                if sim < min_sim:
                    min_sim = sim
                pre_wrd = section_data[i]

        return min_sim


    def calc_ratio_of_avg_sim_to_final_sim(self, data):
        return self.calc_avg_sim(data) / self.calc_final_sim(data)

    def plot_avg_sim_to_sim_ratio(self):
        avg_sims = []
        ratios = []

        for truncation, noise_degree in itertools.product(self.truncations, self.noise_degrees):
            data = self.load_data(truncation, noise_degree)
            avg_sim = self.calc_avg_sim(data)
            ratio = self.calc_ratio_of_avg_sim_to_final_sim(data)
            avg_sims.append(avg_sim)
            ratios.append(ratio)

        plt.scatter(ratios, avg_sims)
        plt.title("avg sim and the ratio of avg sim to final sim")
        plt.xlabel("ratio of avg sim to final sim")
        plt.ylabel("avg sim")
        plt.grid(True)
        plt.show()

    def plot_min_sim_to_sim_ratio(self):
        min_sims = []
        ratios = []

        for truncation, noise_degree in itertools.product(self.truncations, self.noise_degrees):
            data = self.load_data(truncation, noise_degree)
            min_sim = self.calc_min_sim(data)
            ratio = self.calc_ratio_of_avg_sim_to_final_sim(data)
            min_sims.append(min_sim)
            ratios.append(ratio)

        plt.scatter(ratios, min_sims)
        plt.title("min sim and the ratio of avg sim to final sim")
        plt.xlabel("ratio of avg sim to final sim")
        plt.ylabel("min sim")
        plt.grid(True)
        plt.show()
            


if __name__ == "__main__":
    data_path = "../data/20220924"
    truncations = [0.004, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    noise_degrees = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    analyzer = Analyzer2(data_path, truncations, noise_degrees)

    analyzer.plot_avg_sim_to_sim_ratio()
    analyzer.plot_min_sim_to_sim_ratio()
