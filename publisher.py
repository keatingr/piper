import redis

from utilities import load_image, to_redis

r = redis.Redis(host='localhost', port=6379, db=0)

img = load_image('./images/cat.jpeg')
r.publish("test_channel", to_redis(img))

