"""
 class to handle word2vec
"""
from asyncio.windows_events import NULL
import gensim
import numpy as np

class Word2vec_handler(object):
    def __init__(self, all_wrds):
        """
        init the word2vec handler
        :param all_wrds: [str]
        """
        self.all_wrds = all_wrds

        print("launching word2vec model ....................................")
        self.model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/onuma/Documents/research/program/GoogleNews-vectors-negative300.bin', binary=True)
        print("launched word2vec model!")

        self.available_wrd_idx = []   # [int] (index of available word)
        self.available_wrds = []      # [str] (available word)
        self.wrd_vec_list = [ [] for _ in range(1000)]          # [float] (word vector, unavailable word vector is NULL)
        self.unavailable_wrd_idxs = [ i for i in range(1000)]  # [int] (index of unabailable word)
        
        self.search_available_word()     

    def search_available_word(self):
        """
        search available word in word2vec 
        """
        for idx, wrd in enumerate(self.all_wrds):
            wrd_list = wrd.split(', ')
            
            for wrd in wrd_list:
                isavailable = False
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
                    success_word = case1
                    isavailable = True
                elif case2 in self.model:
                    success_word = case2
                    isavailable = True
                elif case3 in self.model:
                    success_word = case3
                    isavailable = True
                elif case4 in self.model:
                    success_word = case4
                    isavailable = True
                elif case5 in self.model:
                    success_word = case5
                    isavailable = True
                elif case6 in self.model:
                    success_word = case6
                    isavailable = True
                elif case7 in self.model:
                    success_word = case7
                    isavailable = True
                elif case8 in self.model:
                    success_word = case8
                    isavailable = True
                elif case9 in self.model:
                    success_word = case9
                    isavailable = True
                elif case10 in self.model:
                    success_word = case10
                    isavailable = True
                elif case11 in self.model:
                    success_word = case11
                    isavailable = True

                if isavailable:
                    self.available_wrds.append(success_word)
                    self.available_wrd_idx.append(idx)
                    self.unavailable_wrd_idxs.remove(idx)
                    self.wrd_vec_list[idx] = self.model[success_word]
                    break
        
    def get_cos_sim(self, v1, v2):
        """
        calculate cosine similarity 
        :param v1: np.array (dim 300)
        :param v2: np.array (dim 300)
        :return: float (cosine similarity)
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def make_wrd_vec_file(self):

        for i in range(0, 1000):
            if i in self.unavailable_wrd_idxs:
                self.wrd_vec_list[i] = [0 for _ in range(300)]

        wrd_vec_list = np.array(self.wrd_vec_list)
        unavailable_wrd_idxs = np.array(self.unavailable_wrd_idxs)

        np.savez('word_vector', wrd_vec_list=wrd_vec_list, unavailable_wrd_idxs=unavailable_wrd_idxs)

if __name__ == "__main__":
    with open('imagenet_classes.txt') as f:
        all_wrds = [line.strip() for line in f.readlines()]
    test_class = Word2vec_handler(all_wrds)
    test_class.make_wrd_vec_file()
