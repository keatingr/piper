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
    read_frames = r.xread({'vidstream': b'0-0'})
    read_inference = r.xread({'infstream': b'0-0'})  # number of messages and block count in ms

    vid = []
    for k, item in enumerate(read_frames[0][1]):  # format is tuple of bytes (timestamp, frame data dict) (b'1581893417647-0', {b'frame':b'\x00...blah}
        try:
            img = from_redis(item[1][b'frame'])
        except Exception as e:
            if 'not subscriptable' in str(e):  # TODO figure this out and handle all gracefully the initialization error is for a reason
                pass
            else:
                print("error when calling from_redis(), printing raw item from channel")
                print(e)
            continue

        img = np.squeeze(img)
        vid.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))

        img = img / 255
        timestamp_idx = item[0].decode('utf-8')
        p_label = read_inference.get()[b'prediction']
        if p_label:
            cv2.putText(img, p_label, (11, 21), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 100, 0), 1)
        #
        #
        cv2.imshow('Sporty', img)
        # cv2.moveWindow('Sporty', 300, 150)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    rhost = 'trex'
    r = redis.Redis(host=rhost, port=6379, db=0)
    main()
