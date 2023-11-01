import subprocess
import os
import numpy as np

class Experiment_root_20231022(object):
    def __init__(self, img_diversity_list, wrd_diversity_list, step_num, iteration, memory_limit_step, output_path):
        self.img_diversity_list = img_diversity_list
        self.wrd_diversity_list = wrd_diversity_list
        self.step_num = step_num
        self.iteration = iteration
        self.memory_limit_step = memory_limit_step
        self.output_path = output_path

        self.memory_limit_iter = self.memory_limit_step // self.step_num
        self.section_num, self.rest_iteration = divmod(self.iteration, self.memory_limit_iter)

        self.main()
    
    def main(self):
        for img_diversity in self.img_diversity_list:
            for wrd_diversity in self.wrd_diversity_list:

                case_res = []
                for section_count in range(self.section_num):
                    print(f"----------img div:{img_diversity}, wrd div:{wrd_diversity}, section count:{section_count}----------")
                    section_res = self.exe(img_diversity, wrd_diversity, self.memory_limit_iter, section_count*self.memory_limit_iter)
                    case_res.extend(section_res)
                if self.rest_iteration:
                    print(f"----------img div:{img_diversity}, wrd div:{wrd_diversity}, section count:{self.section_num}----------")
                    section_res = self.exe(img_diversity, wrd_diversity, self.rest_iteration, self.section_num*self.memory_limit_iter)
                    case_res.extend(section_res)
            
                output_path = os.path.join(self.output_path, f"{img_diversity}-{wrd_diversity}.npy")
                np.save(output_path, np.array(case_res))
    
    def exe(self, img_diversity, wrd_diversity, iteration, iter_count):
        stdin = str(img_diversity) + "\n" + str(wrd_diversity) + "\n" + str(self.step_num) + "\n" + str(iteration) + "\n" + str(iter_count) + "\n" + self.output_path
        stdin = stdin.encode()
        section_res = subprocess.run(["python", "experiment_20231022.py"], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        section_res = section_res.stdout.decode()
        section_res = self.decode_section_res(section_res)

        return section_res
    
    def decode_section_res(self, section_res):
        decoded_res = []
        str_flag = False
        quotation_flag = False

        for char in section_res:
            if char == "[":
                one_play_res = []
            elif char == "'" or char == '"':
                if str_flag:
                    quotation_flag = True
                    step_str += char
                else:
                    step_str = ""
                    str_flag = True
            elif quotation_flag:
                if char == ",":
                    one_play_res.append(step_str[:-1])
                    str_flag = False
                elif char == "]":
                    one_play_res.append(step_str[:-1])
                    decoded_res.append(one_play_res)
                    str_flag = False
                else:
                    step_str += char
                quotation_flag = False
            elif str_flag:
                step_str += char

        return decoded_res

if __name__ == "__main__":
    img_diversity_list = [0.04, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    wrd_diversity_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    step_num = 50
    iteration = 50

    memory_limit_step = 500

    output_path = "./data"

    experiment_root = Experiment_root_20231022(img_diversity_list, wrd_diversity_list, step_num, iteration, memory_limit_step, output_path)