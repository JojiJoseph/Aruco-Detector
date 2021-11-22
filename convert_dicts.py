import cv2.aruco as aruco
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm

predefined_dicts = {name:getattr(aruco,name) for name in dir(aruco) if name.startswith("DICT_")}

os.makedirs("./dict", exist_ok=True)

for key in tqdm(predefined_dicts.keys()):
    dict_val = predefined_dicts[key]
    dict = aruco.getPredefinedDictionary(dict_val)
    n_markers = len(dict.bytesList)
    new_dict = {}
    for i in range(n_markers):
        img = dict.drawMarker(i, (dict.markerSize+2))
        img //= 255
        new_dict[tuple(img.ravel())] = (i,0)
        img = np.rot90(img)
        new_dict[tuple(img.ravel())] = (i,1)
        img = np.rot90(img)
        new_dict[tuple(img.ravel())] = (i,2)
        img = np.rot90(img)
        new_dict[tuple(img.ravel())] = (i,3)
    with open(os.path.join("./dict",key + ".pickle"), "wb") as f:
        pickle.dump(new_dict, f)