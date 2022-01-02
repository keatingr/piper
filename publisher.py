# from json import JSONEncoder

import cv2
import numpy as np
import redis

from utilities import to_redis


def main():
    # encoder = JSONEncoder()
    video_file = VIDEO_FILE
    cap = cv2.VideoCapture(video_file)
    if not cap:
        print("No video loaded {}".format(video_file))
        exit()
    while True:
        ret, img = cap.read()
        if not ret:  # end of stream
            break

        img = np.array([img], dtype=np.float32)  # TODO retain uint8 and convert to float32 on the other end to reduce transport cost
        frame = {"frame": to_redis(img)}  # ndarray w x h x 3 uint8
        r.xadd('vidstream', frame)


if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, db=0)
    VIDEO_FILE = './videos/tennis.mp4'
    r.flushall()
    main()
