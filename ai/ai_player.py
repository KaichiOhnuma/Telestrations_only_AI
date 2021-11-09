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
        print('----------sketch by player {} at round {}----------'.format(self.index, round_count))

        imgs = Download_img(word, 10, self.index, round_count).get_imgs()
        result = Compare_img().get_most_similar_img(imgs)
        print("__________{}__________".format(result))
        return result

    def guess(self, sketch, round_count):
        """
        make a guess from the sketch
        :param sketch: str (file path)
        :return: str
        """
        print('----------guess by player {} at round {}----------'.format(self.index, round_count))
        compare_results = []

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\kaich\\Documents\\research\\program\\telestrations-project-395eec6b87fc.json'
        client = vision.ImageAnnotatorClient()
        with open(sketch,'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        labels = response.label_annotations

        for i, label in enumerate(labels):
            label_img_list = Download_img(label.description, 1, self.index, round_count).get_imgs()
            label_img = Compare_img().get_most_similar_img(label_img_list)
            compare_results.append(Compare_img().feature_detection(sketch, label_img))

        result_idx = compare_results.index(min(compare_results))
        result = labels[result_idx].description

        print("__________{}__________".format(result))
        return result

# test
if __name__ == '__main__':
    test = AI_Player(0)
    print(test.guess('C:/Users/kaich/Documents/research/program/Telestrations/ai/images/0-1-0.png', 2))
