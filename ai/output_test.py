import numpy as np
import csv

class Output_data(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.load_file()
        self.output_csv()

    def load_file(self):
        results_file = np.load(self.file_name)
        self.avg_sim_to_p_wrd_list = results_file["avg_sim_to_p_wrd_list"]
        self.avg_sim_to_s_wrd_list = results_file["avg_sim_to_s_wrd_list"]
        self.avg_final_sim_to_s_wrd_list = results_file["avg_final_sim_to_s_wrd_list"]
        self.prob_of_failure_list = results_file["prob_of_failure_list"]
        self.avg_num_of_wrd_list = results_file["avg_num_of_wrd_list"]
        self.setting_list = results_file["setting_list"]

        print(self.avg_sim_to_p_wrd_list)
        print(self.avg_sim_to_s_wrd_list)
        print(self.avg_final_sim_to_s_wrd_list)
        print(self.prob_of_failure_list)
        print(self.avg_num_of_wrd_list)
        print(self.setting_list)

    def output_csv(self):
        with open("C:/Users/onuma/Documents/research/thesis/graduation_thesis/data/result_data1.xlsx") as f:
            writer = csv.writer(f)
            data = self.avg_sim_to_p_wrd_list.tolist()
            for i in range(len(data)):
                data[i] = str(data)
            writer.writerows(data)

if __name__ == "__main__":
    file_name = "result_data1.npz"
    output = Output_data(file_name)