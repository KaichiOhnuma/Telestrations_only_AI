import numpy as np
import pandas as pd

class Output_data(object):
    def __init__(self, file_name):
        self.file_path = "../data/" + file_name
        self.load_file()
        self.output_csv()

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
                self.add_data_csv(data_list, data_list_name)

    def make_csv_file(self):
        with open(self.file_path.replace(".npz", ".csv"), "w+") as f:
            pass

    def add_data_csv(self, data_list, data_list_name):
        output_data = [[truncation] for truncation in self.truncation_list]

        for i, data in enumerate(data_list):
            case = i // len(self.mutation_degree_list)
            output_data[case].append(data)

        columns = [data_list_name]
        columns.extend(self.mutation_degree_list)
        df = pd.DataFrame(output_data, columns=columns)

        df.to_csv(self.file_path.replace(".npz", ".csv"), mode='a', index=False)
        print(df)

if __name__ == "__main__":
    output = Output_data("result_data2.npz")
