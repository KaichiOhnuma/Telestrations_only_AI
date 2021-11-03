"""
get image(s) from google chrome
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

class Get_img_from_google(object):
    def __init__(self, word, n, player_idx, round_count):
        """
        init the object
        :word: str
        :n: int
        :round_count: int
        """
        self.word = word
        self.n = n
        self.player_idx = player_idx
        self.round_count = round_count
        self.images = []
        self.driver = webdriver.Chrome("C:\\Users\\kaich\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\chromedriver\\chromedriver.exe")
        self.image_download()

    def scroll(self):
        last_height = self.driver.execute_script('return document.body.scrollHeight')
        while True:
            self.driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            time.sleep(2)
            new_height = self.driver.execute_script('return document.body.scrollHeight')
            if new_height == last_height:
                break
            last_height = new_height

    def image_download(self):
        """
        get picture from google images-search page and storage to local folder
        :return: None
        """
        self.driver.get("https://www.google.com/search?q="+str(self.word)+"&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjInYnFku_zAhWVxIsBHWlDAjMQ_AUoBHoECAIQBg")
        #self.scroll()
        
        for i in range(self.n):
            try:
                self.driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i+1)+']/a[1]/div[1]/img').screenshot("C:\\Users\\kaich\\Documents\\research\\program\\Telestrations\\ai\\images\\"+str(self.player_idx)+"-"+str(self.round_count)+"-"+str(i)+".png")
                self.images.append("C:\\Users\\kaich\\Documents\\research\\program\\Telestrations\\ai\\images\\"+str(self.player_idx)+"-"+str(self.round_count)+"-"+str(i)+".png")
            except:
                print("error")

        self.driver.close()

    def get_images(self):
        """
        get images list
        :return: [str]
        """
        return self.images

# test
if __name__ == '__main__':
    test = Get_img_from_google("desk", 10, 0, 0)
    print(test.get_images())






'''
#Will keep scrolling down the webpage until it cannot scroll no more
last_height = self.driver.execute_script('return document.body.scrollHeight')
while True:
    self.driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(2)
    new_height = self.driver.execute_script('return document.body.scrollHeight')
    try:
        self.driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
        time.sleep(2)
    except:
        pass
    if new_height == last_height:
        break
    last_height = new_height
'''