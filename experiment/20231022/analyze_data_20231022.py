import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

class Analyze_data_20231022(object):
    def __init__(self, data_path, img_diversity_list, wrd_diversity_list):
        self.data_path = data_path
        self.img_diversity_list = img_diversity_list
        self.wrd_diversity_list = wrd_diversity_list

        self.wrd_vec_data = self.load_wrd_vec_data()

        self.main()

    def main(self):
        avg_success_rate_of_one_step_list = []
        avg_num_of_used_wrd_list = []
        avg_sim_of_one_step_list = []
        avg_final_sim_list = []
        ratio_of__avg_sim_of_one_step__to__avg_final_sim_list = []
        avg_sim_trans_from_first_wrd_list = []

        for img_diversity in self.img_diversity_list:
            for wrd_diversity in self.wrd_diversity_list:
                data = self.load_data(img_diversity, wrd_diversity)
                avg_success_rate_of_one_step = self.get_avg_success_rate_of_one_step(data)
                avg_num_of_used_wrd = self.get_avg_num_of_used_wrd(data)
                avg_sim_of_one_step = self.get_avg_sim_of_one_step(data)
                avg_final_sim = self.get_avg_final_sim(data)
                ratio_of__avg_sim_of_one_step__to__avg_final_sim = self.get_ratio_of__avg_sim_of_one_step__to__avg_final_sim(data)
                avg_sim_trans_from_first_wrd = self.get_avg_sim_trans_from_first_wrd(data)
                

                avg_success_rate_of_one_step_list.append(avg_success_rate_of_one_step)
                avg_num_of_used_wrd_list.append(avg_num_of_used_wrd)
                avg_sim_of_one_step_list.append(avg_sim_of_one_step)
                avg_final_sim_list.append(avg_final_sim)
                ratio_of__avg_sim_of_one_step__to__avg_final_sim_list.append(ratio_of__avg_sim_of_one_step__to__avg_final_sim)
                avg_sim_trans_from_first_wrd_list.append(avg_sim_trans_from_first_wrd)

        self.save_heatmap(avg_success_rate_of_one_step_list, os.path.join(self.data_path, "graphs", "success_rate_of_one_step.png"))
        self.save_heatmap(avg_num_of_used_wrd_list, os.path.join(self.data_path, "graphs", "n_used_wrd.png"))
        self.save_heatmap(avg_sim_of_one_step_list, os.path.join(self.data_path, "graphs", "one_step_sim.png"))
        self.save_heatmap(avg_final_sim_list, os.path.join(self.data_path, "graphs", "final_sim.png"))
        self.save_heatmap(ratio_of__avg_sim_of_one_step__to__avg_final_sim_list, os.path.join(self.data_path, "graphs", "ratio_one_sim_to_final_sim.png"))
        self.save_graph_of_sim_trans_from_first_wrd(avg_sim_trans_from_first_wrd_list, os.path.join(self.data_path, "graphs", "sim_trans_from_first_wrd.png"))

    def save_heatmap(self, analyzed_data, output_path):
        analyzed_data = np.array(analyzed_data)
        analyzed_data = analyzed_data.reshape(len(self.img_diversity_list), len(self.wrd_diversity_list))
        
        plt.figure(figsize=(14,8))
        heatmap = sns.heatmap(analyzed_data, annot=True, cmap="rainbow", fmt="1.2f")
        plt.xlabel("wrd diversity")
        plt.ylabel("img diversity")
        plt.xticks(self.wrd_diversity_list)
        plt.yticks(self.img_diversity_list)
        plt.savefig(output_path)

    def save_graph_of_sim_trans_from_first_wrd(self, analyzed_data, output_path):
        analyzed_data = np.array(analyzed_data)
    
        step = [i for i in range(analyzed_data.shape[1])]

        plt.figure(figsize=(14, 8))
        for i, d in enumerate(analyzed_data):
            img_diversity_idx, wrd_diversity_idx = divmod(i, len(self.wrd_diversity_list))
            img_diversity = self.img_diversity_list[img_diversity_idx]
            wrd_diversity = self.wrd_diversity_list[wrd_diversity_idx]
            color_id = self.get_ratio_of__avg_sim_of_one_step__to__avg_final_sim(self.load_data(img_diversity, wrd_diversity))
            plt.plot(step, d, color=cm.jet(color_id/5.5))

        plt.xlabel("step")
        plt.ylabel("sim to first wrd")
        plt.savefig(output_path)

    def get_avg_sim_of_one_step(self,data):
        iteration = len(data)
        step_num = len(data[0])

        avg_sim = 0

        for one_play_data in data:
            pre_wrd = one_play_data[0]
            for i in range(2, step_num, 2):
                sim = self.calc_cos_sim(pre_wrd, one_play_data[i])
                avg_sim += sim
                pre_wrd = one_play_data[i]

        avg_sim = avg_sim / iteration / (step_num // 2)

        return avg_sim

    def get_avg_final_sim(self,data):
        iteration = len(data)
        avg_final_sim = 0

        for one_play_data in data:
            avg_final_sim += self.calc_cos_sim(one_play_data[0], one_play_data[-1])

        avg_final_sim /= iteration

        return avg_final_sim

    def get_avg_success_rate_of_one_step(self, data):
        iteration = len(data)
        step_num = len(data[0])

        success_num = 0

        for one_play_data in data:
            pre_wrd = one_play_data[0]
            for i in range(2, step_num, 2):
                if pre_wrd == one_play_data[i]:
                    success_num += 1
                pre_wrd = one_play_data[i]

        avg_success_rate = success_num / iteration / (step_num // 2)

        return avg_success_rate


    def get_avg_num_of_used_wrd(self, data):
        iteration = len(data)
        step_num = len(data[0])

        avg_num_of_used_wrd = 0

        for one_play_data in data:
            used_wrd = []
            for i in range(2, step_num, 2):
                if not one_play_data[i] in used_wrd:
                    used_wrd.append(one_play_data[i])
                    avg_num_of_used_wrd += 1

        avg_num_of_used_wrd = avg_num_of_used_wrd / iteration

        return avg_num_of_used_wrd

    def get_ratio_of__avg_sim_of_one_step__to__avg_final_sim(self, data):
        return self.get_avg_sim_of_one_step(data) / self.get_avg_final_sim(data)
    
    def get_avg_sim_trans_from_first_wrd(self, data):
        iteration = len(data)
        step_num = len(data[0])

        avg_sim_trans_from_first_wrd = [0 for _ in range(step_num//2+1)]

        for one_play_data in data:
            first_wrd = one_play_data[0]
            avg_sim_trans_from_first_wrd[0] = 1 * iteration
            for i in range(2, step_num, 2):
                sim_from_first_wrd = self.calc_cos_sim(first_wrd, one_play_data[i])
                avg_sim_trans_from_first_wrd[int(i/2)] += sim_from_first_wrd

        avg_sim_trans_from_first_wrd = [sum_sim/iteration for sum_sim in avg_sim_trans_from_first_wrd]

        return avg_sim_trans_from_first_wrd


    def load_data(self, img_diversity, wrd_diversity):
        data = os.path.join(self.data_path, f"{img_diversity}-{wrd_diversity}.npy")
        data = np.load(data, allow_pickle=True)
        
        return data
    
    def load_wrd_vec_data(self):
        load_wrd_vec = np.load("../../ai/word_vector.npz")
        wrd_vec_list = load_wrd_vec["wrd_vec_list"]
        wrd_vec_data = {}
        with open("../../ai/imagenet_classes.txt") as f:
            for i, line in enumerate(f.readlines()):
                wrd = line.replace("\n", "")
                wrd_vec_data[wrd] = wrd_vec_list[i]

        return wrd_vec_data
    
    def calc_cos_sim(self, w1, w2):
        v1 = self.wrd_vec_data[w1]
        v2 = self.wrd_vec_data[w2]

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    data_path = "./data"
    img_diversity_list = [0.04, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    wrd_diversity_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

    analyze = Analyze_data_20231022(data_path, img_diversity_list, wrd_diversity_list)


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
            