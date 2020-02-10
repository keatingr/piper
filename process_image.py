from keras.applications.vgg16 import (
    VGG16, decode_predictions)

from utilities import load_image

img = load_image('./images/cat.jpeg')
print(img)
model = VGG16(weights='imagenet')
predictions = model.predict(img)
# convert the probabilities to class labels
label = decode_predictions(predictions)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
# print(np.argmax(predictions))