import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt

class Execute_analyzer4(object):
    def __init__(self, data_path, img_diversity, wrd_diveristy):
        self.data_path = data_path
        self.img_diversity = img_diversity
        self.wrd_diversity = wrd_diversity
        self.data = self.load_data()
        self.category_list = ["save_img_dei_scatter", "save_wrd_dei_scatter", "get_avg", "get_variance"]

        self.analyze()

    def analyze(self):
        for category in self.category_list:
            if category == "save_img_dei_scatter":
                x, y = np.array([]), np.array([])
                for section_id in range(len(self.data)):
                    print(f"-----------------img diversity: {self.img_diversity}, wrd diversity: {self.wrd_diversity}, section: {section_id}--------------------")
                    res = self.execute_analyzer4(section_id, category)
                    x = np.append(x, res[0])
                    y = np.append(y, res[1])
                plt.scatter(x, y, vmin=-1, vmax=1, c="red")
                output_path = os.path.join(self.data_path, "analyze", "affect_of_img_dei", f"{self.img_diversity}-{self.wrd_diversity}.png")
                plt.savefig(output_path)
                plt.cla()
            elif category == "save_wrd_dei_scatter":
                x, y = np.array([]), np.array([])
                for section_id in range(len(self.data)):
                    print(f"-----------------img diversity: {self.img_diversity}, wrd diversity: {self.wrd_diversity}, section: {section_id}--------------------")
                    res = self.execute_analyzer4(section_id, category)
                    x = np.append(x, res[0])
                    y = np.append(y, res[1])
                plt.scatter(x, y, vmin=-1, vmax=1, c="blue")
                output_path = os.path.join(self.data_path, "analyze", "affect_of_wrd_dei", f"{self.img_diversity}-{self.wrd_diversity}.png")
                plt.savefig(output_path)
                plt.cla()
            elif category == "get_avg":
                avg_actual_sim, avg_sim_from_img, avg_sim_from_wrd = 0, 0, 0
                for section_id in range(len(self.data)):
                    print(f"-----------------img diversity: {self.img_diversity}, wrd diversity: {self.wrd_diversity}, section: {section_id}--------------------")
                    res = self.execute_analyzer4(section_id, category)
                    avg_actual_sim += res[0]
                    avg_sim_from_img += res[1]
                    avg_sim_from_wrd += res[2]
                avg_actual_sim /= len(self.data)
                avg_sim_from_img /= len(self.data)
                avg_sim_from_wrd /= len(self.data)
                self.avg_actual_sim = avg_actual_sim
                self.avg_sim_from_img = avg_sim_from_img
                self.avg_sim_from_wrd = avg_sim_from_wrd
                print(f"avg actual sim: {self.avg_actual_sim}, avg sim from img: {self.avg_sim_from_img}, avg sim from wrd: {self.avg_sim_from_wrd}")

    def load_data(self):
        data = os.path.join(self.data_path, f"{self.img_diversity}-{self.wrd_diversity}.npy")
        data = np.load(data, allow_pickle=True)
        
        return data
    
    def execute_analyzer4(self, section_id, category):
        stdin = self.data_path + "\n" + str(section_id) + "\n" + str(self.img_diversity) + "\n" + str(self.wrd_diversity) + "\n" + category + "\n"
        stdin = stdin.encode()
        res = subprocess.run(["python", "analyzer4.py"], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        res = res.stdout.decode()

        res = self.read_res(res, category)

        return res

    def read_res(self, res, category):
        if category == "save_img_dei_scatter" or category == "save_wrd_dei_scatter":
            res = res.split("]")
            x = res[0].split("[")
            x = x[2].split(", ")
            y = res[1].split("[")
            y = y[1].split(", ")

            assert len(x) == len(y)
            for i in range(len(x)):
                x[i] = float(x[i])
                y[i] = float(y[i])

            return [x, y]

        elif category == "get_avg":
            res = res.split("[")
            res = res[1].split("]")
            res = res[0].split(", ")
            x = float(res[0])
            y = float(res[1])
            z = float(res[2])

            return [x, y, z] # actual_sim, sim_from_img, sim_from_wrd


if __name__ == "__main__":
    data_path = "../data/20220924"
    case_list = [[0.004, 1.5], [0.004, 0], [1.0, 1.5], [1.0, 2.5], [2.0, 1.5], [2.5, 1.0], [3.0, 2.5], [3.5, 2.5]]
    for case in case_list:
        img_diversity = case[0]
        wrd_diversity = case[1]
        executer = Execute_analyzer4(data_path, img_diversity, wrd_diversity)