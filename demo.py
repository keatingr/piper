import cv2
import numpy as np
import redis
import tensorflow as tf

from utilities import from_redis


def main():
    IMG_HEIGHT = 171  # 171
    IMG_WIDTH = 128  # 128
    FRAME_BATCH_LEN = 16

    # model = tf.keras.models.load_model('./models/sports1m-keras-tf2.h5')
    #
    # model.compile(loss='mean_squared_error', optimizer='sgd')
    #
    # with open('labels.txt', 'r') as f:
    #     labels = [line.strip() for line in f.readlines()]
    # print('Total labels: {}'.format(len(labels)))
    #
    # mean_cube = np.load('models/train01_16_128_171_mean.npy')
    # mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
    # read_frames = r.xread({'vidstream': b'1-0'})
    # read_inference = r.xread({'infstream': b'0-0'})  # number of messages and block count in ms
    # read_frames = r.xrange('vidstream', '-', '+', 5)
    read_inference = r.xrange('infstream', '-', '+', 5)  # number of messages and block count in ms


    vid = []
    for k, item in enumerate(read_inference):
        timestamp_idx = item[1][b'timestamp']
        p_label = item[1][b'prediction']
        print(p_label)
        # img = from_redis(item[1][b'frame'])
        # timestamp_idx = item[0].decode('utf-8')
        #
        # img = np.squeeze(img)
        # vid.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))
        #
        # img = img / 255
        #
        # p_label = read_inference.get(timestamp_idx)
        # if p_label:
        #     cv2.putText(img, p_label, (11, 21), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
        #     cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 100, 0), 1)
        #
        # cv2.imshow('Sporty', img)
        # # cv2.moveWindow('Sporty', 300, 150)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == '__main__':
    try:
        r = redis.StrictRedis(
            host='trex',
            port=6379,
            password='')
        r.ping()
        main()
    except Exception as ex:
        print('Error:', ex)
        exit('Failed to connect')
