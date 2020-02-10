import redis

from utilities import from_redis, predict_image

r = redis.Redis(host='localhost', port=6379, db=0)
sub = r.pubsub()
sub.subscribe("test_channel")
for item in sub.listen():
    try:
        ndarray = from_redis(item['data'])
        prediction = predict_image(ndarray)
        print(prediction)
    except Exception as e:
        print("error when calling from_redis(), printing raw item from channel")
        print(e)
        # print(item)