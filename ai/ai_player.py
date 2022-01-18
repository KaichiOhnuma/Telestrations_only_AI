"""
Represents a AI player object on the client side
"""
import numpy as np
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample, convert_to_images
from torchvision import models, transforms
import torch
from PIL import Image
import gc

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

        wrd_vec_file = np.load("word_vector.npz")
        self.unavailable_wrd_idxs = wrd_vec_file["unavailable_wrd_idxs"]
        self.wrd_vec_list = wrd_vec_file["wrd_vec_list"]

    def sketch(self, wrd, round_count, truncation=1.):
        """
        sketch from the word
        :param wrd: str
        :param round_count: int
        :param truncation: float, defaults to 1. 
        :return: str (image file)
        """
        print(f'sketching by player {self.idx} at round {round_count}....')

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

        del class_vector, noise_vector, output, img
        gc.collect()

        return img_file

    def guess(self, img_file, round_count, mutation_rate, mutation_degree):
        """
        guess the image
        :param img_file: str
        :param round_count: int
        :return: str (word)
        """
        print(f'guessing by player {self.idx} at round {round_count}....')

        wrd_vec_dis = []
        base_vector = np.full(300, 0.0)
        mutation = [True, False]
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

        if np.random.choice(mutation, p=[mutation_rate, 1-mutation_rate]):
            base_vector = base_vector * mutation_degree

        for idx, wrd_vec in enumerate(self.wrd_vec_list):
            if idx in self.unavailable_wrd_idxs:
                wrd_vec_dis.append(-1)
            else:
                d = np.dot(base_vector, wrd_vec) / (np.linalg.norm(base_vector) * np.linalg.norm(wrd_vec))
                wrd_vec_dis.append(d)
        
        res_wrd_idx = wrd_vec_dis.index(max(wrd_vec_dis))
        res_wrd = self.wrds[res_wrd_idx]

        return res_wrd

        
# test
if __name__ == '__main__':
    test = AI_Player(0)
    print(test.sketch("flamingo", round_count=0, truncation=1))
