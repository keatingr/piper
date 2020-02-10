import struct

import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from keras.preprocessing import image


def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img, dtype=np.uint16)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def to_redis(ndarray):
    a, h, w, d = ndarray.shape
    shape = struct.pack('>IIII',a,h,w,d)
    encoded = shape + ndarray.tobytes()
    return encoded

def predict_image(ndarray):
    model = VGG16(weights='imagenet')
    predictions = model.predict(ndarray)
    # convert the probabilities to class labels
    label = decode_predictions(predictions)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    prediction = '%s (%.2f%%)' % (label[1], label[2] * 100)
    return prediction

def from_redis(encoded):
    a, h, w, d = struct.unpack('>IIII',encoded[:16])
    decoded = np.frombuffer(encoded, dtype=np.float32, offset=16).reshape(a, h,w, d)
    return decoded