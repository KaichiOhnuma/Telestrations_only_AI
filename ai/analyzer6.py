import numpy as np

wrd_dict = {}

wrd_vec_file = np.load("word_vector.npz")
wrd_vec_list = wrd_vec_file["wrd_vec_list"]
unavailable_wrd_ids = wrd_vec_file["unavailable_wrd_idxs"]


with open('imagenet_classes.txt') as f:
    for i, line in enumerate(f.readlines()):
        wrd = line.replace("\n", "")
        if not i in unavailable_wrd_ids:
            wrd_dict[wrd] = {"id": i, "vector": wrd_vec_list[i]}
        else:
            wrd_dict[wrd] = {"id": i, "vector": None}

w1 = "liner, ocean liner"
w2 = "schooner"
vec1 = wrd_dict[w1]["vector"]
vec2 = wrd_dict[w2]["vector"]

res = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(res)
