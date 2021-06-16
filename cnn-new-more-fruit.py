# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# du coup ce code fonctionne, en tt cas avec quelques modif et cela a fonctionné ^^

# jai changé le nombre de 60 a 65 qui est le nombre de classes

no_of_classes = 131

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), input_shape=(100, 100, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Adding a third convolutional layer
classifier.add(Conv2D(256, (5, 5), activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=no_of_classes, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

Training = train_datagen.flow_from_directory('fruits-360/Training',
                                             target_size=(100, 100),
                                             batch_size=32,
                                             class_mode='categorical')

Validation = test_datagen.flow_from_directory('fruits-360/Test',
                                              target_size=(100, 100),
                                              batch_size=32,
                                              class_mode='categorical')

history = classifier.fit_generator(Training,
                                   steps_per_epoch=len(Training),
                                   epochs=25,
                                   validation_data=Validation,
                                   validation_steps=len(Validation))
classifier.summary()
#
# # Plotting graphs
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

# creation du dictionaire of the model history methode 2
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)
# Plot the training/validation loss
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Plot the training/validation accuracy
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


classifier.save("dernierModelCnnNewMoreFruit.h5")
print("model créé !")

# 131 fruit
name_of_classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
                   'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red Delicious', 'Apple Red Yellow 1',
                   'Apple Red Yellow 2',
                   'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana',
                   'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit',
                   'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower',
                   'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow',
                   'Cherry 1', 'Cherry 2', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe',
                   'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue',
                   'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4',
                   'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry',
                   'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee',
                   'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry',
                   'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled',
                   'Onion White',
                   'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat', 'Peach 2', 'Pear', 'Pear Abate',
                   'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pear 2',
                   'Pepino',
                   'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk',
                   'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate',
                   'Pomelo Sweetie',
                   'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry',
                   'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato Cherry Red',
                   'Tomato Heart', 'Tomato Maroon', 'Tomato Not Ripened', 'Tomato Yellow', 'Tomato 1', 'Tomato 2',
                   'Tomato 3', 'Tomato 4', 'Walnut', 'Watermelon']

test_image = image.load_img('img/banana.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/banana.jpg'
Image.open(img)

test_image = image.load_img('img/hb.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/hb.jpg'
Image.open(img)

test_image = image.load_img('img/lychee.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/lychee.jpg'
Image.open(img)

test_image = image.load_img('img/coco.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/coco.jpg'
Image.open(img)

test_image = image.load_img('img/gapp2.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/gapp2.jpg'
Image.open(img)

test_image = image.load_img('img/cf.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/cf.jpg'
Image.open(img)

test_image = image.load_img('img/lmn.jpg', target_size=(100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
Training.class_indices
img = 'img/lmn.jpg'
Image.open(img)

# test_image = image.load_img('banana.jpg', target_size=(100, 100))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = classifier.predict(test_image)
# Training.class_indices
# img = 'banana.jpg'
# Image.open(img)

for i in range(no_of_classes):
    if (result[0][i] == 1.0):
        img_path = 'fruits-360/Training' + name_of_classes[i] + '/2_100.jpg'
        print('Predicted:', name_of_classes[i])
Image.open(img_path)
