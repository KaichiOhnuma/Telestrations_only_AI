"""
get image(s) from google chrome
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class Get_image_from_google(object):
    def __init__(self, word, n, index, round_count):
        """
        init the object
        :word: str
        :n: int
        :round_count: int
        """
        self.word = word
        self.n = n
        self.index = index
        self.round_count = round_count
        self.images = []
        self.driver = webdriver.Chrome("C:\\Users\\kaich\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\chromedriver\\chromedriver.exe")
        self.image_download()

    def image_download(self):
        """
        get picture from google images-search page and storage to local folder
        :return: None
        """
        self.driver.get("https://www.google.com/search?q="+str(self.word)+"&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjInYnFku_zAhWVxIsBHWlDAjMQ_AUoBHoECAIQBg")
        
        for i in range(1, self.n+1):
            try:
                self.driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot("C:\\Users\\kaich\\Documents\\research\\program\\Telestrations\\client\\images\\"+str(self.index)+"-"+str(i)+".png")
                self.images.append("C:\\Users\\kaich\\Documents\\research\\program\\Telestrations\\client\\images\\"+str(self.index)+"-"+str(self.round_count)+"-"+str(i)+".png")
            except:
                print("error")

        self.driver.close()

    def get_images(self):
        """
        get images list
        :return: [str]
        """
        return self.images





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