""" 
Handles wordnet
"""

from nltk.corpus import wordnet

class Wordnet(object):
    def __init__(self):
        """
        init the wordnet handler
        """
        self.wrd_dict = {} # {word: [Synset[], level(int)]}

    def get_synset(self, wrd):
        """
        get main synset from word
        :param wrd: str
        :return: Synset[] or None
        """
        synsets = wordnet.synsets(wrd)
        
        try:
            synset = synsets[0]
        except IndexError:
            print("The word '{}' isn't available".format(wrd))
            return None

        self.wrd_dict[wrd] = [synset, self.get_abstraction_level(synset)]
        return synset

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

    def get_wrd_dict(self):
        """
        get word dictionary
        :return: {word: [Synset[], int]}
        """
        return self.wrd_dict

# test
if __name__ == '__main__':
    labels = ["Sports_equipment","Ball","Nature","bat-and-ball_games","Bottle_cap",\
        "Grass","Ball_game","Football","Biome","Baseball"]

    test = Wordnet()
    for label in labels:
        test.get_synset(label)

    print(test.wrd_dict)