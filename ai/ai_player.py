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
    def __init__(self, id):
        """
        init the AI player object
        :param index: int
        :param round_count: int
        """
        self.id = id

        self.reliable_label = 5

        self.gan = BigGAN.from_pretrained('biggan-deep-512')
        self.classifier = models.resnet152(pretrained=True)
        self.classifier.eval()

        if torch.cuda.is_available():
            print("k")
            self.gan.to("cuda")
            self.classifier.to("cuda")

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        self.wrd_dict = {}

        wrd_vec_file = np.load("word_vector.npz")
        wrd_vec_list = wrd_vec_file["wrd_vec_list"]
        self.unavailable_wrd_ids = wrd_vec_file["unavailable_wrd_idxs"]

        with open('imagenet_classes.txt') as f:
            for i, line in enumerate(f.readlines()):
                wrd = line.replace("\n", "")
                if not i in self.unavailable_wrd_ids:
                    self.wrd_dict[wrd] = {"id": i, "vector": wrd_vec_list[i]}
                else:
                    self.wrd_dict[wrd] = {"id": i, "vector": None}

    def sketch(self, wrd, round_count, truncation, noise_degree, iter):
        """
        sketch from the word
        :param wrd: str
        :param round_count: int
        :param truncation: float, defaults to 2. 
        :return: str (image file)
        """
        #print(f'sketching by player {self.idx} at round {round_count}....')

        wrd_id = self.wrd_dict[wrd]["id"]
        class_vector = one_hot_from_int([wrd_id], batch_size=1)
        class_vector = torch.from_numpy(class_vector)
        
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
        noise_vector = torch.from_numpy(noise_vector)

        truncation = truncation / 2

        with torch.no_grad():
            output = self.gan(noise_vector, class_vector, truncation)
        output = output.to("cpu")

        img = convert_to_images(output)
        img_path = f'C:/Users/onuma/Documents/research/code/Telestrations_only_AI/ai/images/{self.id}-{iter}-{truncation}-{noise_degree}-{round_count}.png'
        img[0].save(img_path, quality=95)

        del class_vector, noise_vector, output, img
        gc.collect()

        return img_path

    def guess(self, img_path, noise_rate, noise_degree):
        """
        guess the image
        :param img_file: str
        :param round_count: int
        :return: str (word)
        """
        #print(f'guessing by player {self.idx} at round {round_count}....')
        base_vector = np.full(300, 0.0)

        img = Image.open(img_path)
        img = self.transform(img)
        
        classification = self.classifier(img.unsqueeze(0))
        unavailable_wrd_score = torch.min(classification)
        for unavailable_wrd_idx in self.unavailable_wrd_ids:
            classification[0 ,unavailable_wrd_idx] = unavailable_wrd_score

        percentages = torch.nn.functional.softmax(classification, dim=1)[0]
        #_, rank_idxs = torch.sort(classification, descending=True)

        for val in self.wrd_dict.values():
            if not val["vector"] is None:
                base_vector += val["vector"] * percentages[val["id"]].item()

        base_vector = self.add_noise(base_vector, noise_rate, noise_degree)

        max_dis = None
        for key, val in self.wrd_dict.items():
            if not val["id"] in self.unavailable_wrd_ids:
                dis =  np.dot(base_vector, val["vector"]) / (np.linalg.norm(base_vector) * np.linalg.norm(val["vector"]))
                if max_dis is None or max_dis < dis:
                    res_wrd = key
                    max_dis = dis

        del base_vector, img, classification, unavailable_wrd_score, percentages
        gc.collect()

        return res_wrd
    
    def add_noise(self, vec, noise_rate, noise_degree):
        is_add_noise = [True, False]
        if np.random.choice(is_add_noise, p=[noise_rate, 1-noise_rate]):
            noise_vec = 2 * noise_degree * np.random.rand(len(vec)) - noise_degree
            vec += noise_vec

        return vec
        
# test
if __name__ == '__main__':
    test = AI_Player(0)
    test.sketch("hen", 0, truncation=0.04)
    print(test)

