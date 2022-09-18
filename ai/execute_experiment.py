from distutils.util import execute
import itertools
import subprocess

from experiment import Experiment


class Execute_Experiment(object):
    def __init__(self, truncations, noise_degrees, noise_rates, step_num, iteration, memory_limit_step, output_path):
        self.truncations = truncations
        self.noise_degrees = noise_degrees
        self.noise_rates = noise_rates
        self.step_num = step_num
        self.iteration = iteration
        self.memory_limit_step = memory_limit_step
        self.output_path = output_path

        self.memory_limit_iteration = self.memory_limit_step // self.step_num
        self.iter_num, self.rest_iter_num = divmod(self.iteration, self.memory_limit_iteration)

        self.execute_manage()

    def execute_manage(self):
        for truncation, noise_degree, noise_rate in itertools.product(self.truncations, self.noise_degrees, self.noise_rates):
            res = []
            for iter in range(self.iter_num):
                print(f"--------------------trun: {truncation}, noise: {noise_degree}, iter: {iter}---------------------------")
                case_res = self.execute(truncation, noise_degree, noise_rate, self.memory_limit_iteration, self.memory_limit_iteration*iter)
                res.extend(case_res)
            if self.rest_iter_num:
                case_res = self.execute(truncation, noise_degree, noise_rate, self.rest_iter_num, self.memory_limit_iteration*self.iter_num)
                res.extend(case_res)
            print(res)

        return res

    def execute(self, truncation, noise_degree, noise_rate, iteration, iter_count):
        stdin = str(truncation) + "\n" + str(noise_degree) + "\n" + str(noise_rate) + "\n" + str(iteration) + "\n" + str(iter_count)
        stdin = stdin.encode()
        res = subprocess.run(["python", "experiment.py"], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        res = self.read_res(res)

        return res
    
    def read_res(self, res):
        res = res.stdout.decode()

        list_res = []
        str_flag = False
        sub_list_flag = False

        for char in res:
            if char == "[":
                case_res = []
                sub_list_flag = True
            elif char == "]" and sub_list_flag:
                list_res.append(case_res)
                sub_list_flag = False
            elif char == "'":
                if str_flag:
                    case_res.append(step_str)
                    str_flag = False
                else:
                    step_str = ""
                    str_flag = True
            elif str_flag:
                step_str += char

        return list_res




if __name__ == "__main__":
    truncations = [0.004, 0.5, 1.0, 1.5, 2.0]
    noise_degrees = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    noise_rates = [1]

    step_num = 50
    iteration = 6

    memory_limit_step = 150

    output_path = "20220917.npz"

    experiment = Execute_Experiment(truncations, noise_degrees, noise_rates, step_num, iteration, memory_limit_step, output_path)