"""
Represents the object of image compare function
""" 

import cv2

class Compare_img(object):
    def __init__(self):
        """
        init the object of the img compare function
        """
        self.img_size = (200,200)

    def compare_hist_match(self, target_img, comparing_img):
        """
        compare image by hist match
        :param target_img: str (file path)
        :param comparing_img: str (file path)
        :return: float
        """
        target_img = cv2.imread(target_img)
        target_img = cv2.resize(target_img, self.img_size)
        target_img_hist = cv2.calcHist([target_img], [0], None, [256], [0,256])

        comparing_img = cv2.imread(comparing_img)
        comparing_img = cv2.resize(comparing_img, self.img_size)
        comparing_img_hist = cv2.calcHist([comparing_img], [0], None, [256], [0,256])

        result = cv2.compareHist(target_img_hist, comparing_img_hist, 0)

        return result 

    def feature_detection(self, target_img, comparing_img):
        """
        compare image by feature detection
        :param target_img: str (file path)
        :param comparing_img: str (file path)
        :return: float
        """
        bf = cv2.BFMatcher(cv2.NORM_L2)
        detector = cv2.AKAZE_create()

        try:
            target_img = cv2.imread(target_img, cv2.IMREAD_GRAYSCALE)
            target_img = cv2.resize(target_img, self.img_size)
            (target_img_kp, target_img_des) = detector.detectAndCompute(target_img, None)

            comparing_img = cv2.imread(comparing_img, cv2.IMREAD_GRAYSCALE)
            comparing_img = cv2.resize(comparing_img, self.img_size)
            (comparing_img_kp, comparing_img_des) = detector.detectAndCompute(comparing_img, None)
        except:
            print("Error: image is not available")
            result = 10000

        try:
            matches = bf.match(target_img_des, comparing_img_des)
            dist = [match.distance for match in matches]
            result = sum(dist) / len(dist)
        except:
            print("Error: compare is failed")
            result = 100000

        return result

    def get_most_similar_img(self, imgs):
        """
        get the most similar image
        :param imgs: [str]
        :return: str (file path)
        """
        avg_compare_results = [0 for _ in range(len(imgs))]
        
        for i, target_img in enumerate(imgs):
            for j, comparing_img in enumerate(imgs):
                if i != j:
                    compare = self.feature_detection(target_img, comparing_img)
                    avg_compare_results[i] += compare / (len(imgs) - 1)
        
        result_idx = avg_compare_results.index(min(avg_compare_results))
        result = imgs[result_idx]
        
        return result



# test
if __name__ == '__main__':
    imgs = []
    for i in range(10):
        imgs.append('C:/Users/onuma/Documents/research/program/Telestrations/ai/images/2-1-'+str(i)+'.png')
    test = Compare_img()
    #res_test1 = test.feature_detection(target_img, comparing_img1)
    #res_test2 = test.feature_detection(target_img, comparing_img2)
    res_test3 = test.get_most_similar_img(imgs)

    #print(res_test1)
    #print(res_test2)
    print(res_test3)
    
