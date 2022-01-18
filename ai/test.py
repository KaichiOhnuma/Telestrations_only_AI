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
        self.wrd_books_list = []
        self.truncation_list = [0.004, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.mutation_rate_list = [0]
        self.mutation_degree_list = [1]
        
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
        self.wrd_book = []
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
                self.wrd_book.append(guess)

        self.wrd_books.append(self.wrd_book)
    
    def make_secret_wrd(self):
        """
        make secret word
        """
        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        secret_wrd_idx = np.random.randint(0, 1000)
        while secret_wrd_idx in self.player.word_vector_handler.unavailable_wrd_idxs:
            secret_wrd_idx = np.random.randint(0, 1000)

        self.sketch_book.append(wrds[secret_wrd_idx])
        self.wrd_book.append(wrds[secret_wrd_idx])

    def evaluate_process(self):

        self.one_step_vec_process = []
        self.cos_sim_to_secret_wrd = []
        self.prob_of_failure = []

        for wrd_books in self.wrd_books_list:
            list1 = []
            list2 = []
            failed = 0

            for wrd_book in wrd_books:
                secret_wrd = wrd_book[0]
                secret_wrd_idx = self.player.wrds.index(secret_wrd)
                secret_wrd_vec = self.player.word_vector_handler.wrd_vec_list[secret_wrd_idx]
                previous_wrd_vec = secret_wrd_vec

                list1_1 = []
                list2_1 = []
                
                for current_wrd in wrd_book:
                    current_wrd_idx = self.player.wrds.index(current_wrd)
                    current_wrd_vec = self.player.word_vector_handler.wrd_vec_list[current_wrd_idx]
                    sim_to_s_wrd = self.player.word_vector_handler.get_cos_sim(secret_wrd_vec, current_wrd_vec)
                    sim_to_p_wrd = self.player.word_vector_handler.get_cos_sim(previous_wrd_vec, current_wrd_vec)
                    list1_1.append(sim_to_s_wrd)
                    list2_1.append(sim_to_p_wrd)
                    if sim_to_p_wrd != 1:
                        failed += 1
                
                list1.append(list1_1)
                list2.append(list2_1)

            self.one_step_vec_process.append(list1)
            self.cos_sim_to_secret_wrd.append(list2)
            self.prob_of_failure.append(failed / (self.step_n/2) / self.iteration)

        self.av_sim_one_step = []
        self.av_sim_to_s_wrd = []
        self.final_sim = []
        
        for i in range(len(self.wrd_books_list)):
            mean_one_step = 0
            mean_sim_to_s_wrd = 0
            mean_final_sim = 0

            for j in range(self.iteration):
                mean_one_step += sum(self.one_step_vec_process[i][j]) / len(self.one_step_vec_process[i][j]) / self.iteration
                mean_sim_to_s_wrd += sum(self.cos_sim_to_secret_wrd[i][j]) / len(self.cos_sim_to_secret_wrd[i][j]) / self.iteration
                mean_final_sim += self.cos_sim_to_secret_wrd[i][j][int(self.step_n/2)] / self.iteration
            
            self.av_sim_one_step.append(mean_one_step)
            self.av_sim_to_s_wrd.append(mean_sim_to_s_wrd)
            self.final_sim.append(mean_final_sim)




    def main(self):
        for setting in self.setting:
            truncation = setting[0]
            mutation_rate = setting[1]
            mutation_degree = setting[2]
            self.wrd_books = []

            for i in range(self.iteration):
                print("----------------------truncation: {}, mutation rate: {}, mutation degree: {}, round: {}-------------------------"
                .format(truncation, mutation_rate, mutation_degree, i+1))
                self.round(truncation=truncation, mutation_rate=mutation_rate, mutation_degree=mutation_degree)

            self.wrd_books_list.append(self.wrd_books)

        print(self.wrd_books_list)
        self.evaluate_process()

if __name__ == "__main__":
    step_n = 20
    iteration = 10

    test = Test(step_n=step_n, iteration=iteration)

    for i, setting in enumerate(test.setting):
        print("\n-----------truncation:{}, mutation rate:{}, mutation degree:{}---------------------"
        .format(setting[0], setting[1], setting[2]))
        print("average cosine similarity in one step: {}".format(test.av_sim_one_step[i]))
        print("average cosine similarity to secret word: {}".format(test.av_sim_to_s_wrd[i]))
        print("final cosine similarity to secret word: {}".format(test.final_sim[i]))
        print("probability of failure: {}".format(test.prob_of_failure[i]))