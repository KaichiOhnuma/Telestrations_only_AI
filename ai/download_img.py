"""
Represents img(s) download object
"""

from bs4 import BeautifulSoup
import requests
import json

class Download_img(object):
    def __init__(self, word, n, player_idx, round_count):
        """
        init the img(s) download object
        :param word: str
        :param n: int
        :param player_idx: int
        :param round_count: int
        """
        self.word = word
        self.n = n
        self.player_idx = player_idx
        self.round_count = round_count
        self.imgs = []
        self.download()

    def download(self):
        """
        download img(s)
        :return: None
        """
        page = 'https://www.bing.com/images/search'
        headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0"}
        params = {'q': self.word, 'hl': 'en', 'form': 'HDRSC2', 'first': '1', 'scenario': 'ImageBasicHover'}

        response = requests.get(page, headers=headers,  params=params)
        soup = BeautifulSoup(response.text, "lxml")
        data_list = soup.find_all('a', {'class': 'iusc'})

        for i in range(self.n):
            data = data_list[i]
            json_data = json.loads(data.get('m'))
            img_link = json_data['murl']
            res = requests.get(img_link, stream=True)
            if res.status_code == 200:
                img_file = 'C:/Users/kaich/Documents/research/program/Telestrations/ai/images/'+str(self.player_idx)+'-'+str(self.round_count)+'-'+str(i)+'.png'
                with open(img_file, 'wb') as f:
                    f.write(res.content)
                    self.imgs.append(img_file)
            else:
                print("download is failed")

    def get_imgs(self):
        """
        get downloaded image list 
        :return: [str]
        """
        return self.imgs

# test
if __name__ == '__main__':
    test = Download_img('kidney beans', 10, 0, 0).get_imgs()
    print(test)