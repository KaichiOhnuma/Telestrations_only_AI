'''
Represents a AI player object on the client side
'''
from google.cloud import vision
from get_image_from_google import Get_image_from_google

class AI_Player(object):
    def __init__(self, index):
        '''
        init the AI player object
        :param index: int
        '''
        self.index = index

    def sketch(self, word):
        """
        make a sketch from the word
        :param word: str
        :return: str (file path)
        """
        img = Get_image_from_google(word, 1, self.index).get_images()
        return img

    def guess(self, sketch):
        """
        make a guess from the sketch
        :param sketch: str (file path)
        :return: str
        """
        pass