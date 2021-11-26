""" 
Handles wordnet
"""

from nltk.corpus import wordnet

class Wordnet(object):
    def __init___(self):
        """
        init the wordnet handler
        """
        self.wrd_dict = {}

    def get_synset(self, wrd):
        """
        get main synset from word
        :param wrd: str
        :return: Synset[] or None
        """
        synsets = wordnet.synsets(wrd)
        
        try: 
            return synsets[0]
        except IndexError:
            print("The word isn't available")

    def get_hypernym(self, synset):
        """
        get main hypernym from synset
        :param synset: Synset[]
        :return: Synset[] or None
        """

        try:
            hypernyms = synset.hypernyms()
        except AttributeError:
            print("[AttributeError] Synset is NULL")
            return None

        try:
            return hypernyms[0]
        except IndexError:
            print("[IndexError] Hypernym doesn't exist")
            return None


    def get_abstraction_level(self, synset):
        """
        get abstraction level of synset
        :param synset: Synset[]
        :return: int
        """
        level = 0
        hypernym = self.get_hypernym(synset)

        while hypernym is not None:
            hypernym = self.get_hypernym(hypernym)
            level += 1

        return level

# test
if __name__ == '__main__':
    test = Wordnet()
    synset = test.get_synset('kidney bean')
    hypernym = test.get_hypernym(synset)
    level = test.get_abstraction_level(synset)
    print(synset)
    print(hypernym)
    print(level)
