import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.pubsub()
r.set('foo', 'bar')
response = r.get('foo')
print(response)