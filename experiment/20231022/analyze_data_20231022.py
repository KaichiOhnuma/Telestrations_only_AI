import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import umap
import umap.plot
import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")

from sentence_transformers import SentenceTransformer
import umap
import pandas as pd

class Analyze_data_20231022(object):
    def __init__(self, data_path, img_diversity_list, wrd_diversity_list):
        self.data_path = data_path
        self.img_diversity_list = img_diversity_list
        self.wrd_diversity_list = wrd_diversity_list

        self.wrd_vec_data = self.load_wrd_vec_data()
        self.wrd_vec_model = SentenceTransformer("stsb-xlm-r-multilingual")

        self.main()

    def main(self):
        avg_success_rate_of_one_step_list = []
        avg_num_of_used_wrd_list = []
        avg_sim_of_one_step_list = []
        avg_final_sim_list = []
        ratio_of__avg_sim_of_one_step__to__avg_final_sim_list = []
        avg_sim_trans_from_first_wrd_list = []
        correlation_of_sim_and_synsets_num_list = []
        correlation_of_sim_and_synset_deepness_list = []

        for img_diversity in self.img_diversity_list:
            for wrd_diversity in self.wrd_diversity_list:
                data = self.load_data(img_diversity, wrd_diversity)
                avg_success_rate_of_one_step = self.get_avg_success_rate_of_one_step(data)
                avg_num_of_used_wrd = self.get_avg_num_of_used_wrd(data)
                avg_sim_of_one_step = self.get_avg_sim_of_one_step(data)
                avg_final_sim = self.get_avg_final_sim(data)
                ratio_of__avg_sim_of_one_step__to__avg_final_sim = self.get_ratio_of__avg_sim_of_one_step__to__avg_final_sim(data)
                avg_sim_trans_from_first_wrd = self.get_avg_sim_trans_from_first_wrd(data)
                correlation_of_sim_and_synsets_num = self.get_correlation_of_sim_and_synsets_num(data)
                correlation_of_sim_and_synset_deepness = self.get_correlation_of_sim_and_synset_deepness(data)
                
                # self.save_umap(data, os.path.join(self.data_path, "graphs", "umap", f"{img_diversity}-{wrd_diversity}"))
                # self.save_scatter_of_sim_and_abstraction_level(data, os.path.join(self.data_path, "graphs", "scatter" ,f"{img_diversity}-{wrd_diversity}"))

                avg_success_rate_of_one_step_list.append(avg_success_rate_of_one_step)
                avg_num_of_used_wrd_list.append(avg_num_of_used_wrd)
                avg_sim_of_one_step_list.append(avg_sim_of_one_step)
                avg_final_sim_list.append(avg_final_sim)
                ratio_of__avg_sim_of_one_step__to__avg_final_sim_list.append(ratio_of__avg_sim_of_one_step__to__avg_final_sim)
                avg_sim_trans_from_first_wrd_list.append(avg_sim_trans_from_first_wrd)
                correlation_of_sim_and_synsets_num_list.append(correlation_of_sim_and_synsets_num)
                correlation_of_sim_and_synset_deepness_list.append(correlation_of_sim_and_synset_deepness)

        self.save_heatmap(avg_success_rate_of_one_step_list, os.path.join(self.data_path, "graphs", "success_rate_of_one_step.png"))
        self.save_heatmap(avg_num_of_used_wrd_list, os.path.join(self.data_path, "graphs", "n_used_wrd.png"))
        self.save_heatmap(avg_sim_of_one_step_list, os.path.join(self.data_path, "graphs", "one_step_sim.png"))
        self.save_heatmap(avg_final_sim_list, os.path.join(self.data_path, "graphs", "final_sim.png"))
        self.save_heatmap(ratio_of__avg_sim_of_one_step__to__avg_final_sim_list, os.path.join(self.data_path, "graphs", "ratio_one_sim_to_final_sim.png"))
        self.save_heatmap(correlation_of_sim_and_synsets_num_list, os.path.join(self.data_path, "graphs", "correlation_of_sim_and_synsets_num.png"))
        self.save_heatmap(correlation_of_sim_and_synset_deepness_list, os.path.join(self.data_path, "graphs", "correlation_of_sim_and_synset_deepness.png"))
        self.save_graph_of_sim_trans_from_first_wrd(avg_sim_trans_from_first_wrd_list, os.path.join(self.data_path, "graphs", "sim_trans_from_first_wrd.png"))

    def save_heatmap(self, analyzed_data, output_path):
        analyzed_data = np.array(analyzed_data)

        data_frame = []

        for i, data in enumerate(analyzed_data):
            img_diversity = self.img_diversity_list[i//len(self.wrd_diversity_list)]
            wrd_diversity = self.wrd_diversity_list[i%len(self.wrd_diversity_list)]
            data_frame.append([img_diversity, wrd_diversity, data])

        data_frame = pd.DataFrame(data_frame, columns=["img diversity", "wrd diversity", "target data"])
        data_frame = data_frame.pivot("img diversity", "wrd diversity", "target data")

        plt.figure(figsize=(14,8))
        heatmap = sns.heatmap(data_frame, annot=True, cmap="rainbow", fmt="1.2f")
        plt.xlabel("wrd diversity")
        plt.ylabel("img diversity")
        plt.savefig(output_path)
        plt.clf()
        plt.close()

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
        plt.clf()
        plt.close()

    def save_umap(self, data, output_path):
        step_num = len(data[0])

        wrd_trans = []
        wrd_vec_trans = []

        for i, one_play_data in enumerate(data):
            wrd_trans = [one_play_data[i].split(",")[0] for i in range(0, step_num, 2)]
            wrd_vec_trans = [self.wrd_vec_model.encode(wrd, convert_to_tensor=True).numpy() for wrd in wrd_trans]

            coords = umap.UMAP(n_neighbors=4, n_components=2, min_dist=0.3, metric='cosine', random_state=10).fit_transform(wrd_vec_trans)

            fig, ax = plt.subplots()
            x = [vector[0] for vector in coords]
            y = [vector[1] for vector in coords]
            ax.scatter(x[0], y[0], c="red")
            ax.scatter(x, y, c="blue")
            ax.plot(x, y, c="gray", linewidth=0.3)
            for j, wrd in enumerate(wrd_trans):
                ax.annotate(wrd, (coords[j][0], coords[j][1]), fontsize=10)
            plt.savefig(output_path + f"-{i}.png")
            plt.clf()
            plt.close()

    def save_scatter_of_sim_and_abstraction_level(self, data, output_path):        
        step_num = len(data[0])

        one_step_sim_list = []
        synsets_num_list = []
        synset_deepness_list = []

        for one_play_data in data:
            pre_wrd = one_play_data[0]
            for i in range(2, step_num, 2):
                sim = self.calc_cos_sim(pre_wrd, one_play_data[i])

                if self.get_synsets_num(one_play_data[i]) != 0:
                    one_step_sim_list.append(sim)
                    synsets_num_list.append(self.get_synsets_num(one_play_data[i]))
                    synset_deepness_list.append(self.get_synset_deepness(one_play_data[i]))

                pre_wrd = one_play_data[i]
        
        plt.figure(figsize=(14,8))
        plt.scatter(synsets_num_list, one_step_sim_list)
        plt.xlabel("synsets num")
        plt.ylabel("sim of one step")
        correlation = np.corrcoef(np.array(synsets_num_list), np.array(one_step_sim_list))[0][1]
        plt.title(f"correlation: {correlation}")
        plt.savefig(output_path+"-scatter_of_synsets_num_and_sim.png")
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(14,8))
        plt.scatter(synset_deepness_list, one_step_sim_list)
        plt.xlabel("wrd synset deepness")
        plt.ylabel("sim of one step")
        correlation = np.corrcoef(np.array(synset_deepness_list), np.array(one_step_sim_list))[0][1]
        plt.title(f"correlation: {correlation}")
        plt.savefig(output_path+"-scatter_of_synset_deepness_and_sim.png")
        plt.clf()
        plt.close()

    def get_correlation_of_sim_and_synset_deepness(self, data):
        step_num = len(data[0])

        one_step_sim_list = []
        synset_deepness_list = []

        for one_play_data in data:
            pre_wrd = one_play_data[0]
            for i in range(2, step_num, 2):
                sim = self.calc_cos_sim(pre_wrd, one_play_data[i])

                if self.get_synsets_num(one_play_data[i]) != 0:
                    one_step_sim_list.append(sim)
                    synset_deepness_list.append(self.get_synset_deepness(one_play_data[i]))

                pre_wrd = one_play_data[i]

        correlation = np.corrcoef(np.array(synset_deepness_list), np.array(one_step_sim_list))[0][1]
        return correlation
    
    def get_correlation_of_sim_and_synsets_num(self, data):
        step_num = len(data[0])

        one_step_sim_list = []
        synsets_num_list = []

        for one_play_data in data:
            pre_wrd = one_play_data[0]
            for i in range(2, step_num, 2):
                sim = self.calc_cos_sim(pre_wrd, one_play_data[i])

                if self.get_synsets_num(one_play_data[i]) != 0:
                    one_step_sim_list.append(sim)
                    synsets_num_list.append(self.get_synsets_num(one_play_data[i]))

                pre_wrd = one_play_data[i]

        correlation = np.corrcoef(np.array(synsets_num_list), np.array(one_step_sim_list))[0][1]
        return correlation

    def get_synsets_num(self, wrd):
        return len(wn.synsets(wrd, pos=wn.NOUN))

    def get_synset_deepness(self,wrd):
        if len(wn.synsets(wrd)) == 0:
            return 0
        
        wrd_synset = wn.synsets(wrd)[0]

        count = 0
        while True:
            new_wrd_synsets = wrd_synset.hypernyms()

            if len(new_wrd_synsets) == 0:
                break

            wrd_synset = new_wrd_synsets[0]
            count += 1

        return count

    def get_avg_sim_of_one_step(self, data):
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

    def get_avg_final_sim(self, data):
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
            