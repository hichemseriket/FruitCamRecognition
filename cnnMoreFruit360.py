
# read in libraries
import tensorflow as tf
from tensorflow.keras import backend, models, layers, optimizers
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from IPython.display import display
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
from tensorflow.keras.models import Model

np.random.seed(42)

# Specify the base directory where images are located.
base_dir = 'fruits-360/'
# Specify the traning, validation, and test dirrectories.
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Test')
# Normalize the pixels in the train data images, resize and augment the data.
train_data = ImageDataGenerator(
    rescale=1. / 255,  # The image augmentaion function in Keras
    shear_range=0.2,
    zoom_range=0.2,  # Zoom in on image by 20%
    horizontal_flip=True)  # Flip image horizontally
# Normalize the test data imagees, resize them but don't augment them
test_data = ImageDataGenerator(rescale=1. / 255)
train_generator = train_data.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')
test_generator = test_data.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')

# Load InceptionV3 library
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Always clear the backend before training a model
backend.clear_session()
# InceptionV3 model and use the weights from imagenet
conv_base = InceptionV3(weights='imagenet',  # Useing the inception_v3 CNN that was trained on ImageNet data.
                        include_top=False)

# Connect the InceptionV3 output to the fully connected layers
InceptionV3_model = conv_base.output
pool = GlobalAveragePooling2D()(InceptionV3_model)
dense_1 = layers.Dense(512, activation='relu')(pool)

# je modifi ici le nombre de classes
output = layers.Dense(131, activation='softmax')(dense_1)

# Create an example of the Archictecture to plot on a graph
model_example = models.Model(inputs=conv_base.input, outputs=output)
# plot graph
plot_model(model_example)

# Define/Create the model for training
model_InceptionV3 = models.Model(inputs=conv_base.input, outputs=output)
# Compile the model with categorical crossentropy for the loss function and SGD for the optimizer with the learning
# rate at 1e-4 and momentum at 0.9
model_InceptionV3.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

# Import from tensorflow the module to read the GPU device and then print
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# Execute the model with fit_generator within the while loop utilizing the discovered GPU
import tensorflow as tf

with tf.device("/device:GPU:0"):
    history = model_InceptionV3.fit_generator(
        train_generator,
        epochs=5,
        # jai rajouter les steps pour que le model sache comb1 de batches jai pour l'entrainement
        # steps_per_epoch=67692 // 256,
        # validation_steps=22688 // 256,
        validation_data=test_generator,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)])

# Create a dictionary of the model history
import matplotlib.pyplot as plt

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
# Evaluate the test accuracy and test loss of the model
test_loss, test_acc = model_InceptionV3.evaluate_generator(test_generator)
print('Model testing accuracy/testing loss:', test_acc, " ", test_loss)


# j'essaie d'avoir en sortie un model sauvgarder que je pourrait utlisé ou je veux
model_InceptionV3.save("modelInception.h5")
print("model créé !")


###################################################################################################
# j'ai rajouté le serveur ici pour exploiter le model inception
# import os
# from uuid import uuid4
# import cv2
# from time import sleep
# from flask import Flask, request, render_template, send_from_directory
#
# app = Flask(__name__)
# # app = Flask(__name__, static_folder="images")
#
#
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
#            'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
#            'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana',
#            'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit',
#            'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower',
#            'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow',
#            'Cherry 1', 'Cherry 2', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe',
#            'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue',
#            'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4',
#            'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry',
#            'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee',
#            'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry',
#            'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White',
#            'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat', 'Peach 2', 'Pear', 'Pear Abate',
#            'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pear 2', 'Pepino',
#            'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk',
#            'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2',  'Plum 3', 'Pomegranate', 'Pomelo Sweetie',
#            'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry',
#            'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato Cherry Red',
#            'Tomato Heart', 'Tomato Maroon', 'Tomato Not Ripened', 'Tomato Yellow', 'Tomato 1', 'Tomato 2',
#            'Tomato 3', 'Tomato 4', 'Walnut', 'Watermelon']
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# # faut que ici ou je fait appel a la camera et reconnaitre en live
# # sinon je doit au moins faire appel a la capture puis enregistrer la photo que jutiliserais au process
# @app.route("/upload", methods=["POST"])
# def upload():
#     target = os.path.join(APP_ROOT, 'images/')
#     print(target)
#     if not os.path.isdir(target):
#         os.mkdir(target)
#     else:
#         print("Couldn't create upload directory: {}".format(target))
#     print(request.files.getlist("file"))
#     for upload in request.files.getlist("file"):
#         print(upload)
#         print("{} is the file name".format(upload.filename))
#         filename = upload.filename
#         destination = "/".join([target, filename])
#         print("Accept incoming file:", filename)
#         print("Save it to:", destination)
#         upload.save(destination)
#         # import tensorflow as tf
#         import numpy as np
#         from keras.preprocessing import image
#
#         from keras.models import load_model
#
#         # mon model de reconnaissance entrainer que sur 65 classe model.h5
#         # new_model = load_model('model.h5')
#
#         # ce model est entrainer sur 131 classe j'ai mis le serveur ici pour pouvoir utilisé le model
#         # entrainer plus haut qui ne sort pas comme model.h5
#         new_model = load_model(model_InceptionV3)
#
#         # mon autre model entrainer sur 13 classes
#         # new_model = load_model('fruits_fresh_cnn_1.h5')
#
#         # ici jessaye de mettre le model que jai entrainé avec le tuto qui a plus de fruit
#         # new_model = load_model('fruits_fresh_cnn_1.h5')
#
#         new_model.summary()
#         test_image = image.load_img('images\\' + filename, target_size=(64, 64))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis=0)
#         result = new_model.predict(test_image)
#         result1 = result[0]
#         for i in range(6):
#
#             if result1[i] == 1.:
#                 break
#         prediction = classes[i]
#
#     # return send_from_directory("images", filename, as_attachment=True)
#     return render_template("template.html", image_name=filename, text=prediction)
#
#
# @app.route('/upload/<filename>')
# def send_image(filename):
#     return send_from_directory("images", filename)
#
#
# if __name__ == "__main__":
#     app.run(debug=False)
#
    ########################################################################################

