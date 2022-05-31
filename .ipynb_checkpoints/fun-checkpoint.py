# fazer upload da imagem
import sys
import os
import cv2
import numpy as np
from dados_sys import *

from typing import Any
from types import MethodType



color_space = {"GRAY": cv2.COLOR_BGR2GRAY,
               "YCrCb": cv2.COLOR_BGR2YCrCb,
               "HSV": cv2.COLOR_BGR2HSV,
               "HLS": cv2.COLOR_BGR2HLS,
               "Lab": cv2.COLOR_BGR2Lab,}

space_color = {"GRAY": cv2.COLOR_GRAY2BGR,
               "YCrCb": cv2.COLOR_YCrCb2BGR,
               "HSV": cv2.COLOR_HSV2BGR,
               "HLS": cv2.COLOR_HLS2BGR,
               "Lab": cv2.COLOR_Lab2BGR,}

sliding = {"ZigZag" : None,
           "L2R": None,
           "R2L": None}


aut = dict(
    Floyd_steinberg = dict(
        array = np.array([[   0,    1, 7/16],
                          [3/16, 5/16, 1/16]]),
        val = [1,1]
    ),
    Stevenson_arce = dict(
        array = np.array([[     0,     0,     0,     1,     0,32/200,     0],
                          [12/200,     0,26/200,     0,30/200,     0,16/200],
                          [     0,12/200,     0,26/200,     0,12/200,     0],
                          [ 5/200,     0,12/200,     0,12/200,     0, 5/200]]),
        val = [3,3]
    ),
    Burkes = dict(
        array = np.array([[   0,    0,    1, 8/32, 4/32],
                          [2/32, 4/32, 8/32, 4/32, 2/32]]),
        val   = [1,2],
    ),
    Sierra = dict(
        array = np.array([[   0,    0,    1, 5/32, 3/32],
                          [2/32, 4/32, 5/32, 4/32, 2/32],
                          [   0, 2/32, 3/32, 2/32,    0]]),
        val   = [2,2],
    ),
    Stucki = dict(
        array = np.array([[   0,    0,    1, 8/42, 4/42],
                          [2/42, 4/42, 8/42, 4/42, 2/42],
                          [1/42, 2/42, 4/42, 2/42, 1/42]]),
        val   = [2,2],
    ),
    Jarvis = dict(
        array = np.array([[   0,    0,    1, 7/48, 5/48],
                          [3/48, 5/48, 7/48, 5/48, 3/48],
                          [1/48, 3/48, 5/48, 3/48, 1/48]]),
        val   = [2,2],
    ),
)

def load_img(nimg : str) -> Any:
    img = Imagem()
    img.imag_up(nimg)
    return img
    
def meios_tons(array, nome: str = "Floyd_steinberg", mode: str = "L2R"):
    # print("Shape", array.shape)
    g = np.zeros(array.shape)
    h, w = array.shape
    # print(nome)
    a, b = aut[nome]["val"]
    tipo = aut[nome]["array"]
    # print(tipo, a,b)
    invert = list(range(b, w - b))
    
    for y in range(h - a):
        for x in invert:
            g[y, x] = 0 if array[y, x] < 128 else 255
            erro = array[y,x] - g[y,x]
            win = array[y:y+a+1, x-b:x+b+1]
            array[y:y+a+1, x-b:x+b+1] = win + np.round(tipo*erro)
        if mode == "ZigZag":
            invert.reverse()
            tipo = tipo[:, ::-1]
            # print(tipo)
    # cv2.imshow("difusao de erro banda monocromatica", g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return g

def gray(array, aut, mode):
    array = cv2.cvtColor(array,
                         color_space["GRAY"])
    array = meios_tons(array, aut, mode)
    # array = cv2.cvtColor(array,
    #                      space_color["GRAY"])
    return array

def bgr(array, aut, mode):
    array[:,:,0] = meios_tons(array[:,:,0], aut, mode)
    array[:,:,1] = meios_tons(array[:,:,1], aut, mode)
    array[:,:,2] = meios_tons(array[:,:,2], aut, mode)
    return array

def hsv(array, aut, mode):
    array    = cv2.cvtColor(array,
                            color_space["HSV"])
    array[:,:,2] = meios_tons(array[:,:,2], aut, mode)
    array    = cv2.cvtColor(array,
                            space_color["HSV"])
    return array

def lab(array, aut, mode):
    array    = cv2.cvtColor(array,
                            color_space["Lab"])
    array[:,:,0] = meios_tons(array[:,:,0], aut, mode)
    array    = cv2.cvtColor(array,
                            space_color["Lab"])
    return array

def plot_img(original, trans, pt):
    if pt == "cv2":
        cv2.imshow("Original", original)
        cv2.imshow("difusao de erro em cada banda", trans)
        cv2.imshow("difusao de erro banda monocromatica", trans[:,:,2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if pt == "plt":
        fig, axs = plt.subplots(figsize=(10,10))
        axs.set_title("Original")
        # axs.append( fig.add_subplot(1, 3, 0) )
        axs.imshow(original)
        axs.set_title("difusão de erro em cada banda")
        axs.imshow(trans)
        axs.set_title("difusão de erro banda monocromatica")
        axs.imshow(trans[:,:,2])
        plt.show()

def process(img_name : str = "baboon.png",
            autor_name:str="Floyd_steinberg",
            mode:str="L2R",
            cor_spec:str="BGR",
            imgout_name:str="imagem_fil.png",
            pt:str=None):
    
    # print(img_name)
    img_original = load_img(img_name)
    new_img      = func[cor_spec](img_original.array.copy(),
                                  autor_name,
                                  mode)
    if pt == "cv2":
        cv2.imshow("original", img_original.array)
        cv2.imshow(imgout_name, new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite("results/"+imgout_name, new_img)
    print("La imagen se encuentra en {}".format("results/"+imgout_name))
    
    

func = {"filtrar": process,
        "GRAY": gray,
        "BGR": bgr,
        "HSV":hsv,
        "Lab":lab}