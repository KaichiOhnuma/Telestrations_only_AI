"""
Represents a AI player object on the client side
"""
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

        img = Image.open(img_file)
        img = self.transform(img)

        classification = self.classifier(img.unsqueeze(0))
        _, rank_idxs = torch.sort(classification, descending=True)
        percentage = torch.nn.functional.softmax(classification, dim=1)[0]

        res_idxs = np.array([idx for idx in rank_idxs[0][:self.reliable_label]])
        res_wrds = np.array([self.wrds[idx] for idx in res_idxs])
        res_probs = np.array([percentage[idx].item() for idx in res_idxs])

        print(res_wrds[0])
        return res_wrds[0]

# test
if __name__ == '__main__':
    test = AI_Player(0)
    print(test.sketch('triceratops', 0))
