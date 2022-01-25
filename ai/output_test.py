import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Output_data(object):
    def __init__(self, file_name):
        self.file_path = "../data/" + file_name
        self.load_file()
        #self.output_csv()
        #self.make_heat_map(1, 1)
        self.save_all_heatmap()

        # data1 / data2
        #self.save_extend_heatmap(1, 'mean cosine similarity to previous word', 'mean cosine similarity between secret word and finally delivered word')

    def load_file(self):
        self.results_file = np.load(self.file_path)
        self.avg_sim_to_p_wrd_list = self.results_file["avg_sim_to_p_wrd_list"]
        self.avg_sim_to_s_wrd_list = self.results_file["avg_sim_to_s_wrd_list"]
        self.avg_final_sim_to_s_wrd_list = self.results_file["avg_final_sim_to_s_wrd_list"]
        self.prob_of_failure_list = self.results_file["prob_of_failure_list"]
        self.avg_num_of_wrd_list = self.results_file["avg_num_of_wrd_list"]
        self.setting_list = self.results_file["setting_list"]

        self.truncation_list = []
        self.mutation_degree_list = []
        self.mutation_rate_list = []

        for setting in self.setting_list:
            if not setting[0] in self.truncation_list:
                self.truncation_list.append(setting[0])
            if not setting[1] in self.mutation_degree_list:
                self.mutation_degree_list.append(setting[1])
            if not setting[2] in self.mutation_rate_list:
                self.mutation_rate_list.append(setting[2])

    def output_csv(self):
        self.make_csv_file()

        for mutation_rate in self.mutation_rate_list:
            for i in range(5):
                if i == 0:
                    data_list = self.avg_sim_to_p_wrd_list
                    data_list_name = "avg_sim_to_p_wrd"
                elif i == 1:
                    data_list = self.avg_sim_to_s_wrd_list
                    data_list_name = "avg_sim_to_s_wrd"
                elif i == 2:
                    data_list = self.avg_final_sim_to_s_wrd_list
                    data_list_name = "avg_final_sim_to_s_wrd"
                elif i == 3:
                    data_list = self.prob_of_failure_list
                    data_list_name = "prob_of_failure_list"
                elif i == 4:
                    data_list = self.avg_num_of_wrd_list
                    data_list_name = "avg_num_of_wrd_list" 
                data_list_name += ", mutation rate: " + str(mutation_rate)
                self.add_data_csv(data_list, data_list_name, mutation_rate)

    def make_csv_file(self):
        with open(self.file_path.replace(".npz", ".csv"), "w+") as f:
            pass

    def add_data_csv(self, data_list, data_list_name, mutation_rate):
        output_data = [[truncation] for truncation in self.truncation_list]
        output_data = self.make_csv_output_data(data_list, mutation_rate, output_data)

        columns = [data_list_name]
        columns.extend(self.mutation_degree_list)
        df = pd.DataFrame(output_data, columns=columns)

        df.to_csv(self.file_path.replace(".npz", ".csv"), mode='a', index=False)
        print(df)

    def make_csv_output_data(self, data_list, mutation_rate, output_data):
        output_data_list = []
        mutation_rate_idx = self.mutation_rate_list.index(mutation_rate)
        
        for i, data in enumerate(data_list):
            case = i % len(self.mutation_rate_list)
            if case == mutation_rate_idx:
                output_data_list.append(data)

        for i, data in enumerate(output_data_list):
            case = i // len(self.mutation_degree_list)
            output_data[case].append(data)

        return output_data

    def make_heat_map(self, mutation_rate, data_idx):

        if data_idx == 0:
            data_name = 'mean cosine similarity to previous word'
        elif data_idx == 1:
            data_name = 'mean cosine similarity to secret word'
        elif data_idx == 2:
            data_name = 'mean cosine similarity between secret word and finally delivered word'
        elif data_idx == 3:
            data_name = 'probability of failure at one step'
        else:
            data_name = 'mean the number of delivered words'

        df = self.make_heatmap_data()
        df = df.loc[df['add noise rate'] == mutation_rate]

        output = df.pivot('threshold', '$\u03b1$', data_name)

        plt.figure(figsize=(7, 4))
        heat_map = sns.heatmap(output, annot=True, cmap="coolwarm", fmt="1.2f")
        #heat_map.set_title(data_name)
        file_name = "../data/" + self.file_path.replace(".npz", "-") + str(mutation_rate) + "-" + data_name + ".png"
        plt.savefig(file_name)

    def make_heatmap_data(self):
        output_data = []

        for i, setting in enumerate(self.setting_list):
            data = [setting[0]]
            data.append(setting[1])
            data.append(setting[2])
            data.append(self.avg_sim_to_p_wrd_list[i])
            data.append(self.avg_sim_to_s_wrd_list[i])
            data.append(self.avg_final_sim_to_s_wrd_list[i])
            data.append(self.prob_of_failure_list[i])
            data.append(self.avg_num_of_wrd_list[i])
            output_data.append(data)

        columns = ['threshold', '$\u03b1$', 'add noise rate', 'mean cosine similarity to previous word',
         'mean cosine similarity to secret word', 'mean cosine similarity between secret word and finally delivered word', 
         'probability of failure at one step', 'mean the number of delivered words']
        df = pd.DataFrame(output_data, columns=columns)

        return df

    def save_all_heatmap(self):

        for mutation_rate in self.mutation_rate_list:
            for i in range(5):
                self.make_heat_map(mutation_rate, i)

    def save_extend_heatmap(self, mutation_rate, data1_name, data2_name):
        df = self.make_heatmap_data()
        df = df.loc[df['add noise rate'] == mutation_rate]
        df[data1_name] /= df[data2_name]

        output = df.pivot('threshold', '$\u03b1$', data1_name)
        data_name = data1_name + "per" + data2_name

        plt.figure(figsize=(7, 4))
        heat_map = sns.heatmap(output, annot=True, cmap="coolwarm", fmt="1.2f")
        #heat_map.set_title(data_name)
        file_name = "../data/" + self.file_path.replace(".npz", "-") + str(mutation_rate) + "-" + data_name + ".png"
        plt.savefig(file_name)


if __name__ == "__main__":
    output = Output_data("result_data1.npz")