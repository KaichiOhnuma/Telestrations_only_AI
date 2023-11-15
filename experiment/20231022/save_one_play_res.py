import tkinter
from PIL import Image, ImageTk
import os
import numpy as np

import tkcap

def save_one_play_result(data_path, img_diversity, wrd_diversity, set_count):
    window = tkinter.Tk()
    window.geometry("1680x1260")
    window.configure(bg="white")

    output_path = os.path.join(data_path, "one_play_res", f"{img_diversity}-{wrd_diversity}-{set_count}.png")

    data_path = os.path.join(data_path, f"{img_diversity}-{wrd_diversity}.npy")
    data = np.load(data_path, allow_pickle=True)
    data = data[set_count]

    step_num = len(data)

    imgs_for_memory = []

    for i, d in enumerate(data):
        img_size = (180, 180)
        arrow_space_width = 30

        canvas = tkinter.Canvas(window, bg="#eee", height=img_size[1], width=img_size[0])
        canvas.place(x=(i%8)*(img_size[0]+arrow_space_width), y=(i//8)*img_size[1])

        if i % 2 == 0:
            wrd = d.split(",")[0]
            canvas.create_text(img_size[0]/2, img_size[1]/2, text=wrd, width=180, font=("Times New Roman", 15))
        else:
            img = Image.open(d)
            img = img.resize(img_size)
            img = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=img, anchor=tkinter.NW)

            imgs_for_memory.append(img)

        if i == step_num - 1:
            break

        canvas = tkinter.Canvas(window, bg="white", height=img_size[0], width=arrow_space_width)
        canvas.place(x=(i%8)*(img_size[0]+arrow_space_width)+img_size[0], y=(i//8)*img_size[1])
        canvas.create_text(arrow_space_width/2, img_size[1]/2, text="→", font=("Times New Roman", 20))

    cap = tkcap.CAP(window)
    cap.capture(output_path)

if __name__ == "__main__":
    data_path = "./data"
    img_diversity, wrd_diversity, set_count = 0.04, 0.5, 4
    save_one_play_result(data_path, img_diversity, wrd_diversity, set_count)