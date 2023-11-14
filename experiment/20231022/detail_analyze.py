import numpy as np
import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")

import tkinter
from PIL import Image, ImageTk
import tkcap
import os
import time

def pick_up_many_synsets_wrd_patern(data_path):
    data = load_data(data_path)

    step_num = len(data[0])

    for i, one_play_data in enumerate(data):
        for step_count in range(0, step_num, 2):
            if len(wn.synsets(one_play_data[step_count], pos=wn.NOUN)) >= 5 and one_play_data[step_count] != one_play_data[step_count+2]:
                output_path = data_path.replace("data", "data/example").replace(".npy", f"-{i}-{step_count}.png")
                if  os.path.exists(output_path):
                    break
                save_one_message(*one_play_data[step_count:step_count+3], output_path)

def save_one_message(wrd1, img_path, wrd2, output_path):
    window = tkinter.Tk()
    window.geometry("600x180")
    window.configure(bg="white")

    synset_num = len(wn.synsets(wrd1, pos=wn.NOUN))
    cos_sim = calc_cos_sim(wrd1, wrd2)
    window.title(f"synset num: {synset_num}, cos sim: {cos_sim}")

    img_size = (180, 180)
    arrow_space_width = 30

    canvas = tkinter.Canvas(window, bg="#eee", height=img_size[1], width=img_size[0])
    canvas.place(x=0, y=0)
    wrd = wrd1.split(",")[0]
    canvas.create_text(img_size[0]/2, img_size[1]/2, text=wrd, width=180, font=("Times New Roman", 15))

    canvas = tkinter.Canvas(window, bg="white", height=img_size[0], width=arrow_space_width)
    canvas.place(x=img_size[0], y=0)
    canvas.create_text(arrow_space_width/2, img_size[1]/2, text="→", font=("Times New Roman", 20))

    canvas = tkinter.Canvas(window, bg="#eee", height=img_size[1], width=img_size[0])
    canvas.place(x=(img_size[0]+arrow_space_width), y=0)
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img, anchor=tkinter.NW)

    canvas = tkinter.Canvas(window, bg="white", height=img_size[0], width=arrow_space_width)
    canvas.place(x=(2*img_size[0]+arrow_space_width), y=0)
    canvas.create_text(arrow_space_width/2, img_size[1]/2, text="→", font=("Times New Roman", 20))

    canvas = tkinter.Canvas(window, bg="#eee", height=img_size[1], width=img_size[0])
    canvas.place(x=2*(img_size[0]+arrow_space_width), y=0)
    wrd = wrd2.split(",")[0]
    canvas.create_text(img_size[0]/2, img_size[1]/2, text=wrd, width=180, font=("Times New Roman", 15))

    cap = tkcap.CAP(window)
    cap.capture(output_path)

def load_data(data_path):
    return np.load(data_path, allow_pickle=True)

def calc_cos_sim(w1, w2):
    wrd_vec_data = load_wrd_vec_data()
    v1 = wrd_vec_data[w1]
    v2 = wrd_vec_data[w2]

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def load_wrd_vec_data():
    load_wrd_vec = np.load("../../ai/word_vector.npz")
    wrd_vec_list = load_wrd_vec["wrd_vec_list"]
    wrd_vec_data = {}
    with open("../../ai/imagenet_classes.txt") as f:
        for i, line in enumerate(f.readlines()):
            wrd = line.replace("\n", "")
            wrd_vec_data[wrd] = wrd_vec_list[i]

    return wrd_vec_data

if __name__ == "__main__":
    data_path = "./data/0.75-1.5.npy"
    pick_up_many_synsets_wrd_patern(data_path)