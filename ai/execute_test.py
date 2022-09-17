import subprocess
import numpy as np
import time
import matplotlib.pyplot as plt

class Execute_test(object):
    def __init__(self, step_n, iteration, truncation_list, mutation_degree_list, mutation_rate_list, memory_limit_step, file_name):
        self.step_n = step_n
        self.iteration = iteration
        self.truncation_list = truncation_list
        self.mutation_degree_list = mutation_degree_list
        self.mutation_rate_list = mutation_rate_list
        self.memory_limit_step = memory_limit_step
        self.file_path = "../data/" + file_name

        self.setting_list = []
        
        for truncation in self.truncation_list:
            for mutation_degree in self.mutation_degree_list:
                for mutation_rate in self.mutation_rate_list:
                    self.setting_list.append([truncation, mutation_degree, mutation_rate])
        
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.set_xlabel("time")
        self.ax1.set_ylabel("cosine similarity")
        
        self.execute2()
        #self.print_results()
        #self.save_results(self.file_path)
        self.ax1.legend()
        plt.show()

    def execute(self):
        self.avg_sim_to_p_wrd_list = []
        self.avg_sim_to_s_wrd_list = []
        self.avg_final_sim_to_s_wrd_list = []
        self.prob_of_failure_list = []
        self.avg_num_of_wrd_list = []

        for setting in self.setting_list:
            truncation = setting[0]
            mutation_degree = setting[1]
            mutation_rate = setting[2]

            avg_sim_to_p_wrd = 0
            avg_sim_to_s_wrd = 0
            avg_final_sim_to_s_wrd = 0
            prob_of_failure = 0
            avg_num_of_wrd = 0

            memory_limit_iteration = self.memory_limit_step // self.step_n
            section_num, rest_iteration = divmod(self.iteration , memory_limit_iteration)
            if rest_iteration > 0:
                section_num += 1

            for i in range(section_num):
                print("truncation: {}, mutation rate: {}, mutation degree: {}, section: {} ...."
                .format(truncation, mutation_rate, mutation_degree, i+1))

                if i == section_num - 1 and rest_iteration > 0:
                    res = self.section(rest_iteration, truncation, mutation_degree, mutation_rate)
                else:
                    res = self.section(memory_limit_iteration, truncation, mutation_degree, mutation_rate)

                avg_sim_to_p_wrd += res[0] / section_num
                avg_sim_to_s_wrd += res[1] / section_num
                avg_final_sim_to_s_wrd += res[2] / section_num
                prob_of_failure += res[3] / section_num
                avg_num_of_wrd += res[4] / section_num

            self.avg_sim_to_p_wrd_list.append(avg_sim_to_p_wrd)
            self.avg_sim_to_s_wrd_list.append(avg_sim_to_s_wrd)
            self.avg_final_sim_to_s_wrd_list.append(avg_final_sim_to_s_wrd)
            self.prob_of_failure_list.append(prob_of_failure)
            self.avg_num_of_wrd_list.append(avg_num_of_wrd)

            print(avg_sim_to_p_wrd)
            print(avg_sim_to_s_wrd)
            print(avg_final_sim_to_s_wrd)
            print(prob_of_failure)
            print(avg_num_of_wrd)

            time.sleep(5)

    def execute2(self):

        for setting in self.setting_list:
            truncation, noise_degree, noise_rate = setting
            sim_to_secret_word = None

            memory_limit_iteration = self.memory_limit_step // self.step_n
            section_num, rest_iteration = divmod(self.iteration, memory_limit_iteration)
            if rest_iteration > 0:
                section_num += 1

            for i in range(section_num):
                print(f"truncation: {truncation}, noise rate: {noise_rate}, noise degree: {noise_degree}, section: {i+1} ...")
                if i == section_num -1 and rest_iteration > 0:
                    res = self.section(rest_iteration, truncation, noise_degree, noise_rate)
                else:
                    res = self.section(memory_limit_iteration, truncation, noise_degree, noise_rate)
                if sim_to_secret_word is None:
                    sim_to_secret_word = res / section_num
                else:
                    sim_to_secret_word += res / section_num
            
            t = [i for i in range(len(sim_to_secret_word))]
            self.ax1.plot(t, sim_to_secret_word, label=f"truncation:{truncation*2}, noise:{noise_degree}")

    
    def str_to_list(self, res_str):
        res_list = res_str.split("[")[1].split("]")[0].split()
        for i in range(len(res_list)):
            res_list[i] = float(res_list[i])
        
        return np.array(res_list)
    
    def section(self, iteration, truncation, mutation_degree, mutation_rate):
        stdin = str(self.step_n) + "\n" + str(iteration) + "\n" + str(truncation) + "\n" + str(mutation_degree) + "\n" + str(mutation_rate)
        stdin = stdin.encode()
        res = subprocess.run(["python", "test.py"], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        res_str = res.stdout.decode()
        res_list = self.str_to_list(res_str)

        return res_list

    def print_results(self):
        for i, setting in enumerate(self.setting_list):
            print("------------------truncation: {}, mutation degree: {}, mutation rate: {}----------------------"
            .format(setting[0], setting[1], setting[2]))
            print("average similarity to previous word: {}".format(self.avg_sim_to_p_wrd_list[i]))
            print("average similarity to secret word: {}".format(self.avg_sim_to_s_wrd_list[i]))
            print("average final similarity to secret word: {}".format(self.avg_final_sim_to_s_wrd_list[i]))
            print("probability of failure: {}".format(self.prob_of_failure_list[i]))
            print("average number of word: {}".format(self.avg_num_of_wrd_list[i]))

    def save_results(self, file_name):
        np.savez(file=file_name, avg_sim_to_p_wrd_list=self.avg_sim_to_p_wrd_list, avg_sim_to_s_wrd_list=self.avg_sim_to_s_wrd_list, 
        avg_final_sim_to_s_wrd_list=self.avg_final_sim_to_s_wrd_list, prob_of_failure_list=self.prob_of_failure_list, 
        avg_num_of_wrd_list=self.avg_num_of_wrd_list, setting_list=self.setting_list)





if __name__ == "__main__":
    step_n = 50
    iteration = 50
    truncation_list = [0.002, 0.25, 0.5, 0.75, 1.0]
    mutation_degree_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    mutation_rate_list = [1.0]
    memory_limit_step = 500
    file_name = "20220621.npz"
    main = Execute_test(step_n, iteration, truncation_list, mutation_degree_list, mutation_rate_list, memory_limit_step, file_name)