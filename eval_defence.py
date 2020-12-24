import keras
import sys
import numpy as np
from PIL import Image

img_filename = str(sys.argv[1])
badnet_filename = str(sys.argv[2])


def load_img(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert("RGB")
    img = img.resize((47, 55))
    return np.asarray(img) / 255


def do_eval():
    x = np.array([load_img(img_filename)])
    model_bd = keras.models.load_model("models/"+badnet_filename+".h5")
    model_defence = keras.models.load_model("models/"+badnet_filename+"_defence.h5")
    # number of classes
    N = model_bd.output.shape[1]
    y_bd = np.argmax(model_bd.predict(x), axis=1)[0] + 1
    y_defence = np.argmax(model_defence.predict(x), axis=1)[0] + 1
    if y_defence != y_bd:
        y_defence = N + 1
    print('The class is:', y_defence)


if __name__ == '__main__':
    do_eval()
