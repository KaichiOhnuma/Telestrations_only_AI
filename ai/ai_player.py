"""
Represents a AI player object on the client side
"""
from download_img import Download_img
from compare_img import Compare_img
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
        imgs = Download_img(word, 10, self.index, round_count).get_imgs()
        img_idx = Compare_img(imgs).get_most_similar_img_idx()
        print('sketch by player {} at round {}'.format(self.index, round_count))
        return imgs[img_idx]

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

# test
if __name__ == '__main__':
    test = AI_Player(0)
    print(test.sketch('desk', 0))
