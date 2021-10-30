"""
Represents a AI player object on the client side
"""
from get_image_from_google import Get_image_from_google
import os
from google.cloud import vision

class AI_Player(object):
    def __init__(self, index):
        """
        init the AI player object
        :param index: int
        :param round_count: int
        """
        self.index = index

    def sketch(self, word, round_count):
        """
        make a sketch from the word
        :param word: str
        :return: str (file path)
        """
        img = Get_image_from_google(word, 1, self.index, round_count).get_images()
        return img[0]

    def guess(self, sketch):
        """
        make a guess from the sketch
        :param sketch: str (file path)
        :return: str
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\kaich\\Documents\\research\\program\\telestrations-project-395eec6b87fc.json'
        dir = sketch
        client = vision.ImageAnnotatorClient()
        with open(dir,'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        labels = response.label_annotations
        return labels[0].description