import cv2
import numpy as np
import redis
import tensorflow as tf
from utilities import from_redis, to_redis


def main():
    IMG_HEIGHT = 171  # 171
    IMG_WIDTH = 128  # 128
    FRAME_BATCH_LEN = 16

    model = tf.keras.models.load_model('./models/sports1m-keras-tf2.h5')

    model.compile(loss='mean_squared_error', optimizer='sgd')

    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

    read_samples = r.xread({'vidstream': b'0-0'})  # number of messages and block count in ms
    # print(read_samples)

    vid = []
    for k, item in enumerate(read_samples[0][1]):  # format is tuple of bytes (timestamp, frame data dict) (b'1581893417647-0', {b'frame':b'\x00...blah}
        try:
            img = from_redis(item[1][b'frame'])
        except Exception as e:
            if 'not subscriptable' in str(e):  # TODO figure this out and handle all gracefully the initialization error is for a reason
                pass
            else:
                print("error when calling from_redis(), printing raw item from channel")
                print(e)
                # print(item['frame'])
            continue

        img = np.squeeze(img)
        vid.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))

        if len(vid) == FRAME_BATCH_LEN:
            X = vid - mean_cube  # TODO mean avg is very important!
            X = X[:, 8:120, 30:142, :]  # (l, h, w, c)  # TODO center crop is very important! try without it!
            p = model.predict(np.array([X]))  # TODO can just use X?
            confidence = max(p[0])
            if confidence > 0.2:  # re-label only if thresh
                p_label = '{:.5f} - {}'.format(max(p[0]), labels[int(np.argmax(p[0]))])
                data = {"timestamp": item[0], "prediction": p_label}
                r.xadd('infstream', data)
            vid.pop(0)

        img = img / 255


        # if p_label:
        #     cv2.putText(img, p_label, (11, 21), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
        #     cv2.putText(img, p_label, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 100, 0), 1)
        #
        # cv2.imshow('Sporty', img)
        # # cv2.moveWindow('Sporty', 300, 150)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == '__main__':
    # r = redis.Redis(host='localhost', port=6379, db=0)
    r = redis.Redis('trex')
    r.delete('infstream')
    main()
