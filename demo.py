import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import sleep
import copy
from c3d_model import build_model
import time



def main():
    IMG_HEIGHT = 171  # 171
    IMG_WIDTH = 128  # 128
    START_FRAME = 1
    FRAME_BATCH_LEN = 16
    video_file = VIDEO_FILE

    model = tf.keras.models.load_model('./models/sports1m-keras-tf2.h5')

    model.compile(loss='mean_squared_error', optimizer='sgd')

    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

    cap = cv2.VideoCapture(video_file)
    if not cap:
        print("No video loaded {}".format(video_file))
        exit()
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:  # end of stream
            break
        img = np.array(img, dtype=np.float32)
        vid.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))
        X = vid
        p_label = ''
        if len(vid) == FRAME_BATCH_LEN:
            X -= mean_cube  # TODO mean avg is very important!
            X = X[:, 8:120, 30:142, :]  # (l, h, w, c)  # TODO center crop is very important! try without it!
            # start = time.time()
            p = model.predict(np.array([X]))  # TODO can just use X?
            # print(time.time() - start)
            confidence = max(p[0])
            if confidence > 0.2:  # re-label only if thresh
                p_label = '{:.5f} - {}'.format(max(p[0]), labels[int(np.argmax(p[0]))])
            vid.pop(0)

        img = img / 255
        if p_label:
            cv2.putText(img, p_label, (11, 21), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 100, 0), 1)
        cv2.imshow('Sporty', img)
        cv2.moveWindow('Sporty', 300, 150)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    VIDEO_FILE = './videos/tennis.mp4'
    main()
