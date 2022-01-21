import  sys
import numpy as np

def start():
    x = int(input("x"))
    y = int(input("y"))
    return [x, y]

def load_file():
    results_file = np.load("result_data1.npz")
    avg_sim_to_p_wrd_list = results_file["avg_sim_to_p_wrd_list"]
    avg_sim_to_s_wrd_list = results_file["avg_sim_to_s_wrd_list"]
    avg_final_sim_to_s_wrd_list = results_file["avg_final_sim_to_s_wrd_list"]
    prob_of_failure_list = results_file["prob_of_failure_list"]
    avg_num_of_wrd_list = results_file["avg_num_of_wrd_list"]

    print(avg_sim_to_p_wrd_list)
    print(avg_sim_to_s_wrd_list)
    print(avg_final_sim_to_s_wrd_list)
    print(prob_of_failure_list)
    print(avg_num_of_wrd_list)

if __name__ == "__main__":
    load_file()