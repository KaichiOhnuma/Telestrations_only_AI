import gensim
import numpy as np

class Word2vec_handler(object):
    def __init__(self):
        print("launching word2vec model ....................................")
        self.model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/onuma/Documents/research/program/GoogleNews-vectors-negative300.bin', binary=True)
        print("launched word2vec model!")
        self.make_wrd_vec_list()

    def make_wrd_vec_list(self):
        self.wrd_vec_list = []

        self.search_available_word()

        for wrd in self.wrds:
            self.wrd_vec_list.append(self.model[wrd])

    def search_available_word(self):

        with open('imagenet_classes.txt') as f:
            wrds = [line.strip() for line in f.readlines()]

        self.wrd_idx = []
        self.wrds = []

        for idx, wrd in enumerate(wrds):
            wrd_list = wrd.split(', ')
            for wrd in wrd_list:

                case1 = wrd
                case2 = wrd.replace(' ', '_')
                case3 = wrd.replace('-', '_')
                case4 = wrd.replace('-', '')
                case5 = wrd.replace(' ', '')
                case6 = wrd.capitalize()
                case7 = wrd.replace('-', '').capitalize()
                case8 = wrd.replace(' ', '').capitalize()
                case9 = wrd.replace('-', '').replace(' ', '_')
                case10 = wrd.replace('-', '_').replace(' ', '_') 
                case11 = wrd.replace(' ', '_').lower()

                if case1 in self.model:
                    self.wrds.append(case1)
                    self.wrd_idx.append(idx)               
                    break
                elif case2 in self.model:
                    self.wrds.append(case2)
                    self.wrd_idx.append(idx)               
                    break
                elif case3 in self.model:
                    self.wrds.append(case3)
                    self.wrd_idx.append(idx)               
                    break
                elif case4 in self.model:
                    self.wrds.append(case4)
                    self.wrd_idx.append(idx)               
                    break
                elif case5 in self.model:
                    self.wrds.append(case5)
                    self.wrd_idx.append(idx)               
                    break
                elif case6 in self.model:
                    self.wrds.append(case6)
                    self.wrd_idx.append(idx)               
                    break
                elif case7 in self.model:
                    self.wrds.append(case7)
                    self.wrd_idx.append(idx)               
                    break
                elif case8 in self.model:
                    self.wrds.append(case8)
                    self.wrd_idx.append(idx)               
                    break
                elif case9 in self.model:
                    self.wrds.append(case9)
                    self.wrd_idx.append(idx)               
                    break
                elif case10 in self.model:
                    self.wrds.append(case10)
                    self.wrd_idx.append(idx)               
                    break
                elif case11 in self.model:
                    self.wrds.append(case11)
                    self.wrd_idx.append(idx)               
                    break
        
    def get_cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

if __name__ == "__main__":
    test_class = Word2vec_handler()
    v1 = test_class.wrd_vec_list[644]
    v2 = test_class.wrd_vec_list[642]
    print(test_class.get_cos_sim(v1, v2))
    print(f"{test_class.wrds[644]}-{test_class.wrds[642]}")
