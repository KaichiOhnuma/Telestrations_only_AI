"""
Represents a AI player object on the client side
"""
from asyncio.windows_events import NULL
from word2vec_handler import Word2vec_handler

import numpy as np
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample, convert_to_images
from torchvision import models, transforms
import torch
from PIL import Image

class AI_Player(object):
    def __init__(self, idx):
        """
        init the AI player object
        :param index: int
        :param round_count: int
        """
        self.idx = idx
        self.reliable_label = 5
        self.gan = BigGAN.from_pretrained('biggan-deep-512')
        self.classifier = models.resnet152(pretrained=True)
        self.classifier.eval()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        with open('imagenet_classes.txt') as f:
            self.wrds = [line.strip() for line in f.readlines()]

        self.word_vector_handler = Word2vec_handler(self.wrds)
        self.unavailable_wrd_idxs = self.word_vector_handler.unavailable_wrd_idxs
        self.wrd_vec_list = self.word_vector_handler.wrd_vec_list

    def sketch(self, wrd, round_count, truncation=1.):
        """
        sketch from the word
        :param wrd: str
        :param round_count: int
        :param truncation: float, defaults to 1. 
        :return: str (image file)
        """
        print(f'----------sketch by player {self.idx} at round {round_count}----------')

        wrd_idx = self.wrds.index(wrd)
        class_vector = one_hot_from_int([wrd_idx], batch_size=1)
        class_vector = torch.from_numpy(class_vector)
        
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
        noise_vector = torch.from_numpy(noise_vector)

        with torch.no_grad():
            output = self.gan(noise_vector, class_vector, truncation)

        img = convert_to_images(output)
        img_file = f'C:/Users/onuma/Documents/research/program/Telestrations/ai/images/{self.idx}-{round_count}.png'
        img[0].save(img_file, quality=95)

        print(img_file)
        return img_file

    def guess(self, img_file, round_count):
        """
        guess the image
        :param img_file: str
        :param round_count: int
        :return: str (word)
        """
        print(f'----------guess by player {self.idx} at round {round_count}----------')

        base_vector = np.full(300, 0.0)

        img = Image.open(img_file)
        img = self.transform(img)

        classification = self.classifier(img.unsqueeze(0))
        unavailable_wrd_score = torch.min(classification)
        for unavailable_wrd_idx in self.unavailable_wrd_idxs:
            classification[0 ,unavailable_wrd_idx] = unavailable_wrd_score

        percentages = torch.nn.functional.softmax(classification, dim=1)[0]
        #_, rank_idxs = torch.sort(classification, descending=True)

        for idx, percentage in enumerate(percentages):
            if not idx in self.unavailable_wrd_idxs:
                base_vector += self.wrd_vec_list[idx] * percentage.item()

        return base_vector

        
# test
if __name__ == '__main__':
    test = AI_Player(0)
    print(test.guess("C:/Users/onuma/Documents/research/program/Telestrations/ai/images/0-0.png", 0))
