"""
Represents the object of image compare function
""" 

import cv2

class Compare_img(object):
    def __init__(self, imgs):
        """
        init the object of the img compare function
        :param imgs: [int]
        """
        self.imgs = imgs
        self.img_size = (200,200)

    def compare_hist_match(self):
        """
        compare image(s) with hist match
        :return: [int] (average hist match)
        """
        self.avg_hist_match = [0 for _ in range(len(self.imgs))]

        for i, img1 in enumerate(self.imgs):
            img1 = cv2.imread(img1)
            img1 = cv2.resize(img1, self.img_size)
            img1_hist = cv2.calcHist([img1], [0], None, [256], [0,256])

            for j, img2 in enumerate(self.imgs):
                if i != j:
                    img2 = cv2.imread(img2)
                    img2 = cv2.resize(img2, self.img_size)
                    img2_hist = cv2.calcHist([img2], [0], None, [256], [0,256])

                    result = cv2.compareHist(img1_hist, img2_hist, 0)
                    self.avg_hist_match[i] += result / (len(self.imgs)-1)

        return self.avg_hist_match  

    def feature_detection(self):
        """
        compare image(s) with feature detection
        :return: [int] (average feature distance)
        """
        self.avg_feature_distance = [0 for _ in range(len(self.imgs))]
        bf = cv2.BFMatcher(cv2.NORM_L2)
        detector = cv2.AKAZE_create()

        for i, img1 in enumerate(self.imgs):
            remove_idx_list = []

            img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, self.img_size)
            (img1_kp, img1_des) = detector.detectAndCompute(img1, None)

            for j, img2 in enumerate(self.imgs):
                if i != j:
                    result = 0

                    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.resize(img2, self.img_size)
                    (img2_kp, img2_des) = detector.detectAndCompute(img2, None)
                    
                    try:
                        matches = bf.match(img1_des, img2_des)
                    except:
                        remove_idx_list.append(j)

                    dist = [match.distance for match in matches]
                    result = sum(dist) / len(dist)
                    self.avg_feature_distance[i] += result
                
            for remove_idx in reversed(remove_idx_list):
                self.imgs.pop(remove_idx)
                self.avg_feature_distance.pop(remove_idx)
                
        return self.avg_feature_distance

    def get_most_similar_img_idx(self):
        """
        get the most similar image index
        :return: int
        """
        compare = self.feature_detection()
        min = compare[0]
        img_idx = 0
        
        for i, x in enumerate(compare):
            if x < min:
                min = x
                img_idx = i
        
        return img_idx



# test
if __name__ == '__main__':
    imgs = []
    for i in range(9):
        imgs.append('C:/Users/kaich/Documents/research/program/Telestrations/ai/images/1-1-'+str(i)+'.png')
    test = Compare_img(imgs)
    img_idx = test.get_most_similar_img_idx()
    print(img_idx)
    print(test.imgs[img_idx])
    print(test.avg_feature_distance)