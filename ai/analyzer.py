from re import S
from sre_constants import SUCCESS
from turtle import color
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

class Analyzer(object):
    def __init__(self, truncations, noise_degrees):
        self.truncations = truncations
        self.noise_degrees = noise_degrees

        self.wrd_vec_data = {}

        load_wrd_vec = np.load("word_vector.npz")
        wrd_vec_list = load_wrd_vec["wrd_vec_list"]

        with open("imagenet_classes.txt") as f:
            for i, line in enumerate(f.readlines()):
                wrd = line.replace("\n", "")
                self.wrd_vec_data[wrd] = wrd_vec_list[i]

        self.fig = plt.figure(figsize=(20, 10))
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.set_xlabel("step")
        self.ax1.set_ylabel("cosine similarity")
        # self.ax1.set_ylim(-0.2, 1.1)

    def plot_data(self, data, truncation, noise_degree, color_barometer):
        step = [i for i in range(len(data))]
        # self.ax1.plot(step, data, label=f"pic:{truncation}, wrd:{noise_degree}", color=cm.jet(truncation/self.truncations[-1]))
        # self.ax1.plot(step, data, label=f"pic:{truncation}, wrd:{noise_degree}", color=cm.jet(noise_degree/self.noise_degrees[-1]))
        # self.ax1.plot(step, data, label=f"img:{truncation}, wrd:{noise_degree}")
        self.ax1.plot(step, data, label=f"img:{truncation}, wrd:{noise_degree}", color=cm.jet(color_barometer))

    
    def analysis_manage(self, data_path):
        for truncation, noise_degree in itertools.product(self.truncations, self.noise_degrees):
            data = self.load_data(data_path, truncation, noise_degree)
            success_rate, single_sim, final_sim, sim_ratio = self.make_heatmap_data(data)
            case_res = self.align_data(data)
            self.plot_data(case_res, truncation, noise_degree, single_sim/final_sim/5)
        # plt.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0, fontsize=10)
        output_path = os.path.join(data_path, "sim_transition.png")
        plt.savefig(output_path)
        # plt.show()

    def show_ex(self, data_path, truncation, noise_degree, idx):
        data = self.load_data(data_path, truncation, noise_degree)
        data = data[idx]

    def make_heatmap(self, data_path):
        df = []

        for truncation, noise_degree in itertools.product(self.truncations, self.noise_degrees):
            data = self.load_data(data_path, truncation, noise_degree)
            success_rate, single_sim, final_sim, sim_ratio = self.make_heatmap_data(data)
            case_df = [truncation, noise_degree, success_rate, single_sim, final_sim, sim_ratio]
            df.append(case_df)
        
        df_columns = ["img diversity", "wrd diversity", "success rate", "single sim", "final sim", "sim ration"]
        df = pd.DataFrame(df, columns=df_columns)

        heatmaps = [
            df.pivot(df_columns[0], df_columns[1], df_columns[2]), 
            df.pivot(df_columns[0], df_columns[1], df_columns[3]), 
            df.pivot(df_columns[0], df_columns[1], df_columns[4]), 
            df.pivot(df_columns[0], df_columns[1], df_columns[5])
        ]
        
        for i, hm in enumerate(heatmaps):
            plt.figure(figsize=(7,4))
            heatmap = sns.heatmap(hm, annot=True, cmap="coolwarm", fmt="1.2f")
            output_path = os.path.join(data_path, df_columns[i+2]+".png")
            plt.savefig(output_path)




    def make_heatmap_data(self, data):
        iteration, step_num = data.shape
        success_rate, single_sim, final_sim = 0, 0, 0

        for section_data in data:
            final_sim += self.calc_cos_sim(section_data[0], section_data[-1])
            secret_wrd = section_data[0]
            prev_wrd = secret_wrd
            for step, d in enumerate(section_data):
                if step % 2 == 0 and step != 0:
                    single_sim += self.calc_cos_sim(prev_wrd, d)
                    if prev_wrd == d:
                        success_rate += 1
                    prev_wrd = d
        
        success_rate /= iteration * (step_num // 2)
        single_sim /= iteration * (step_num // 2)
        final_sim /= iteration
        sim_ratio = single_sim / final_sim

        return success_rate, single_sim, final_sim, sim_ratio


    def load_data(self, data_path, truncation, noise_degree):
        data = os.path.join(data_path, f"{truncation}-{noise_degree}.npy")
        data = np.load(data, allow_pickle=True)
        
        return data

    def align_data(self, data):
        iteration, step_num = data.shape

        res = np.array([0.0 for _ in range(step_num//2+1)])

        for section_data in data:
            sim_to_s_wrd = np.array([])
            secret_wrd = section_data[0]
            for step, d in enumerate(section_data):
                if step % 2 == 0:
                    sim = self.calc_cos_sim(secret_wrd, d)
                    sim_to_s_wrd = np.append(sim_to_s_wrd, sim)
            res += sim_to_s_wrd
        
        res /= iteration

        return res

    def calc_cos_sim(self, wrd1, wrd2):
        v1 = self.wrd_vec_data[wrd1]
        v2 = self.wrd_vec_data[wrd2]

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    truncations = [0.004, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    noise_degrees = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    data_path = "../data/20220924"

    analyzer = Analyzer(truncations, noise_degrees)
    analyzer.analysis_manage(data_path)
    # analyzer.make_heatmap(data_path)
    # analyzer.show_ex(data_path, 0.5, 1.5, 0)
    