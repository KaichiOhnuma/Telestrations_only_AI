import requests
import bs4

class Download_img(object):
    def __init__(self, word, n, player_idx, round_count):
        self.word = word
        self.n = n
        self.player_idx = player_idx
        self.round_count = round_count
        self.imgs = []
        self.download()

    def download(self):
        page = requests.get("https://www.google.com/search?hl=en&q="+str(self.word)+"&btnG=Google+Search&tbs=0&safe=off&tbm=isch")
        html = page.text
        soup = bs4.BeautifulSoup(html, 'lxml')
        links = soup.find_all("img")
        for i in range(self.n):
            link = links[i+1].get("src")
            r = requests.get(link, stream=True)
            if r.status_code == 200:
                img = 'C:/Users/kaich/Documents/research/program/Telestrations/ai/images/'+str(self.player_idx)+'-'+str(self.round_count)+'-'+str(i)+'.png'
                with open(img, 'wb') as f:
                    f.write(r.content)
                    self.imgs.append(img)

    def get_imgs(self):
        return self.imgs

# test
if __name__ == '__main__':
    test = Download_img('kidney beans', 10, 0, 0).get_imgs()
    print(test)
