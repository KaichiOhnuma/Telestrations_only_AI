"""
Represent object of Evaluation test for AI player
"""

from ai_player import AI_Player

import numpy as np

class Test(object):
    def __init__(self, step_n, iteration):
        """
        init the Evaluation test object
        :param step_n: int 
        :param iteration: int 
        """
        self.step_n = step_n
        self.iteration = iteration
        self.player = AI_Player(0)
        self.passed_wrd_list_book = []

        wrd_vec_file = np.load("word_vector.npz")
        self.unavailable_wrd_idxs = wrd_vec_file["unavailable_wrd_idxs"]
        self.wrd_vec_list = wrd_vec_file["wrd_vec_list"]

        self.truncation_list = [0.5]
        self.mutation_rate_list = [1]
        self.mutation_degree_list = [2.0]
        
        self.setting = []

        for truncation in self.truncation_list:
            for mutation_rate in self.mutation_rate_list:
                for mutation_degree in self.mutation_degree_list:
                    self.setting.append([truncation, mutation_rate, mutation_degree])

        self.main()

    def round(self, truncation, mutation_rate, mutation_degree):
        """
        do one round
        :return: None
        """
        self.sketch_book = []
        self.passed_wrd = []
        self.make_secret_wrd()

        for i in range(self.step_n):

            # sketch turn
            if i % 2 == 0:
                sketch = self.player.sketch(self.sketch_book[i], round_count=i+1, truncation=truncation)
                self.sketch_book.append(sketch)

            # guess turn
            else:
                guess = self.player.guess(self.sketch_book[i], round_count=i+1, mutation_rate=mutation_rate, mutation_degree=mutation_degree)
                self.sketch_book.append(guess)
                self.passed_wrd.append(guess)

        self.passed_wrd_list.append(self.passed_wrd)
    
    def make_secret_wrd(self):
        """
        make secret word
        """
        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        secret_wrd_idx = np.random.randint(0, 1000)
        while secret_wrd_idx in self.unavailable_wrd_idxs:
            secret_wrd_idx = np.random.randint(0, 1000)

        self.sketch_book.append(wrds[secret_wrd_idx])
        self.passed_wrd.append(wrds[secret_wrd_idx])

    def evaluate_process(self):

        self.sim_to_previous_wrd = []
        self.sim_to_secret_wrd = []

        self.avg_sim_to_p_wrd = []
        self.avg_sim_to_s_wrd = []
        self.avg_final_sim_to_s_wrd = []
        self.prob_of_failure = []

        for passed_wrd_list in self.passed_wrd_list_book:
            sim_to_p_wrd_list = []
            sim_to_s_wrd_list = []
            failed = 0

            for passed_wrd in passed_wrd_list:
                secret_wrd = passed_wrd[0]
                secret_wrd_idx = self.player.wrds.index(secret_wrd)
                secret_wrd_vec = self.wrd_vec_list[secret_wrd_idx]
                previous_wrd_vec = secret_wrd_vec
                previous_wrd_idx = secret_wrd_idx

                sim_to_p_wrd = []
                sim_to_s_wrd = []
                
                for current_wrd in passed_wrd:
                    current_wrd_idx = self.player.wrds.index(current_wrd)
                    current_wrd_vec = self.wrd_vec_list[current_wrd_idx]
                    sim1 = self.get_cos_sim(secret_wrd_vec, current_wrd_vec)
                    sim2 = self.get_cos_sim(previous_wrd_vec, current_wrd_vec)
                    sim_to_p_wrd.append(sim2)
                    sim_to_s_wrd.append(sim1)
                    if previous_wrd_idx != current_wrd_idx:
                        failed += 1

                    previous_wrd_idx = current_wrd_idx
                    previous_wrd_vec = current_wrd_vec
                
                sim_to_p_wrd_list.append(sim_to_p_wrd)
                sim_to_s_wrd_list.append(sim_to_s_wrd)

            self.sim_to_previous_wrd.append(sim_to_p_wrd_list)
            self.sim_to_secret_wrd.append(sim_to_s_wrd_list)

            p = failed / (len(passed_wrd)-1) / self.iteration
            self.prob_of_failure.append(p)
        
        for i in range(len(self.passed_wrd_list_book)):
            avg_sim_to_p_wrd = 0
            avg_sim_to_s_wrd = 0
            avg_final_sim_to_s_wrd = 0

            for j in range(self.iteration):
                avg_sim_to_p_wrd += sum(self.sim_to_previous_wrd[i][j]) / len(self.sim_to_previous_wrd[i][j]) / self.iteration
                avg_sim_to_s_wrd += sum(self.sim_to_secret_wrd[i][j]) / len(self.sim_to_secret_wrd[i][j]) / self.iteration
                avg_final_sim_to_s_wrd += self.sim_to_secret_wrd[i][j][len(self.sim_to_secret_wrd[i][j])-1] / self.iteration
            
            self.avg_sim_to_p_wrd.append(avg_sim_to_p_wrd)
            self.avg_sim_to_s_wrd.append(avg_sim_to_s_wrd)
            self.avg_final_sim_to_s_wrd.append(avg_final_sim_to_s_wrd)

    def get_cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def main(self):

        for setting in self.setting:
            truncation = setting[0]
            mutation_rate = setting[1]
            mutation_degree = setting[2]
            self.passed_wrd_list = []

            for i in range(self.iteration):
                print("----------------------truncation: {}, mutation rate: {}, mutation degree: {}, round: {}-------------------------"
                .format(truncation, mutation_rate, mutation_degree, i+1))
                self.round(truncation=truncation, mutation_rate=mutation_rate, mutation_degree=mutation_degree)

            self.passed_wrd_list_book.append(self.passed_wrd_list)

        print(self.passed_wrd_list_book)
        self.evaluate_process()

if __name__ == "__main__":
    step_n = 50
    iteration = 15

    test = Test(step_n=step_n, iteration=iteration)

    for i, setting in enumerate(test.setting):
        print("\n-----------truncation:{}, mutation rate:{}, mutation degree:{}---------------------"
        .format(setting[0], setting[1], setting[2]))
        print("average cosine similarity to previous word: {}".format(test.avg_sim_to_p_wrd[i]))
        print("average cosine similarity to secret word: {}".format(test.avg_sim_to_s_wrd[i]))
        print("final cosine similarity to secret word: {}".format(test.avg_final_sim_to_s_wrd[i]))
        print("probability of failure: {}".format(test.prob_of_failure[i]))