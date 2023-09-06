import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import itertools




class Execute_analyzer3(object):
    def __init__(self, data_path, all_img_diversity, all_wrd_diversity):
        self.data_path = data_path
        self.all_img_diversity = all_img_diversity
        self.all_wrd_diversity = all_wrd_diversity
        for img_diversity, wrd_diversity in itertools.product(self.all_img_diversity, self.all_wrd_diversity):
            self.execute_manager(img_diversity, wrd_diversity)


    def execute(self, data_path, section_id, img_diversity, wrd_diversity, content):
            stdin = str(data_path) + "\n" + str(section_id) + "\n" + str(img_diversity) + "\n" + str(wrd_diversity) + "\n" + content + "\n"
            stdin = stdin.encode()
            res = subprocess.run(["python", "analyzer3.py"], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            res = self.read_res(res)

            return res

    def execute_manager(self, img_diversity, wrd_diversity):
        data = self.load_data(img_diversity, wrd_diversity)
        x, y = np.array([]), np.array([]) # actual_sim, sim_from_img
        for section_id in range(len(data)):
            print(f"-----------------img diversity: {img_diversity}, wrd diversity: {wrd_diversity}, section: {section_id}--------------------")
            res = self.execute(self.data_path, section_id, img_diversity, wrd_diversity, "wrd_diversity")
            x = np.append(x, res[0])
            y = np.append(y, res[1])

        plt.scatter(x, y, vmin=-1, vmax=1, c="blue")
        output_path = os.path.join(self.data_path, "analyze", "affect_of_wrd_diversity", f"{img_diversity}-{wrd_diversity}.png")
        plt.savefig(output_path)


        x, y = np.array([]), np.array([]) # actual_sim, sim_from_img
        for section_id in range(len(data)):
            print(f"-----------------img diversity: {img_diversity}, wrd diversity: {wrd_diversity}, section: {section_id}--------------------")
            res = self.execute(self.data_path, section_id, img_diversity, wrd_diversity, "img_diversity")
            x = np.append(x, res[0])
            y = np.append(y, res[1])

        plt.scatter(x, y, vmin=-1, vmax=1, c="blue")
        output_path = os.path.join(self.data_path, "analyze", "affect_of_img_diversity", f"{img_diversity}-{wrd_diversity}.png")
        plt.savefig(output_path)

    

    def load_data(self, img_diversity, wrd_diversity):
        data = os.path.join(self.data_path, f"{img_diversity}-{wrd_diversity}.npy")
        data = np.load(data, allow_pickle=True)

        return data

    def read_res(self, res):
        res = res.stdout.decode()
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

if __name__ == "__main__":
    data_path = "../data/20220924"
    all_img_diversity = [0.004, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    all_wrd_diversity = [0, 1.0, 1.5, 2.0, 2.5, 3.0]
    executer = Execute_analyzer3(data_path, all_img_diversity, all_wrd_diversity)