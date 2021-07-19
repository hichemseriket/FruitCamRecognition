import os
from uuid import uuid4
import cv2
from time import sleep
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# les classe avec lesquel jai entrainé le model sur 13 fruit
# classes = ['Apple Braeburn', 'Apple Golden 1', 'Blueberry', 'Cherry 1', 'Cherry 2', 'Fresh Banana', 'Fresh Orange',
#            'Huckleberry', 'Litchi', 'Maracuja', 'Rotten Banana', 'Rotten Orange']
#
# classes = ['apple', 'banana', 'litchi', 'maracuja', 'orange']
# je vais essayé d'entrainné le model sur les data entrainement plus grand en gardant ce script
# oufff enfin j'ai fini de faire les classe jen ai chier a les mettre en ordre et savoir le nombre et ce qui manque
# dans la liste des classe en dur par rapport au dossier  hahaha'
# classes = ['Apple Braeburn', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith',
#            'Apple Red Delicious', 'Apple Red Yellow', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
#            'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Red', 'Cactus fruit', 'Cantaloupe 1',
#            'Cantaloupe 2', 'Carambula', 'Cherry Rainier', 'Cherry 1', 'Cherry 2', 'Clementine', 'Cocos',
#            'Dates', 'Granadilla', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grapefruit Pink', 'Grapefruit White',
#            'Guava', 'Huckleberry', 'Kaki', 'Kiwi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Litchi', 'Mandarine',
#            'Mango', 'Maracuja', 'Melon Piel de Sapo', 'Nectarine', 'Orange', 'Papaya',
#            'Passion Fruit', 'Peach', 'Peach Flat', 'Pear', 'Pear Abate', 'Pear Monster', 'Pear Williams',
#            'Pepino', 'Pineapple', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Quince', 'Raspberry', 'Salak',
#            'Strawberry', 'Tamarillo', 'Tangelo']

# # 131 fruit
classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
           'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
           'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apricot', 'Avocado',
           'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot',
           'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula',
           'Cauliflower', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow',
           'Cherry 1', 'Cherry 2', 'Chestnut', 'Clementine', 'Cocos',
           'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates',
           'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue',
           'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4',
           'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry',
           'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon',
           'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango',
           'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry',
           'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red',
           'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit',
           'Peach', 'Peach Flat', 'Peach 2', 'Pear', 'Pear Abate',
           'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone',
           'Pear Williams', 'Pear 2', 'Pepino', 'Pepper Green', 'Pepper Orange',
           'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple',
           'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2',  'Plum 3',
           'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet',
           'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant',
           'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo',
           'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Not Ripened', 'Tomato Yellow',
           'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Walnut', 'Watermelon']

@app.route("/")
def index():
    return render_template("index.html")

# faut que ici ou je fait appel a la camera et reconnaitre en live
# sinon je doit au moins faire appel a la capture puis enregistrer la photo que jutiliserais au process
@app.route("/upload", methods=["POST"])
def upload():
    # camera = cv2.VideoCapture(0)
    #
    # camera.start_preview(fullscreen=False, window=(50, 50, 640, 480))
    #
    # # un délai est nécessaire pour laisser le temps aux capteurs de se régler
    # sleep(5)
    #
    # # on enregistre le fichier sur le bureau
    # camera.capture('/home/pi/Pictures/image.jpeg')
    #
    # # on fait disparaître l'aperçu.
    # camera.stop_preview()
    global prediction, filename, i
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        # import tensorflow as tf
        import numpy as np
        from keras.preprocessing import image

        from keras.models import load_model

        # mon model de reconnaissance entrainer que sur 65 classe model.h5
        # new_model = load_model('model.h5')

        # ce model est entrainer sur 131 classe
        # new_model = load_model('inceptionDernier.h5')
        new_model = load_model('8juilletMoreSoir.h5')
        # new_model = load_model('dernierModelCnnNewMoreFruit.h5')

        # mon autre model entrainer sur 65 classes
        # new_model = load_model('fruits_fresh_cnn_1.h5')

        # ici jessaye de mettre le model que jai entrainé avec le tuto qui a plus de fruit
        # new_model = load_model('fruits_fresh_cnn_1.h5')
        new_model.summary()
        # pour le model.h5, model5.h5, model13
        # test_image = image.load_img('images\\' + filename, target_size=(64, 64))
        # pour le model13.h5, model65.h5
        test_image = image.load_img('images\\' + filename, target_size=(100, 100))
        #pour les model inception
        # test_image = image.load_img('images\\' + filename, target_size=(299, 299))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        result1 = result[0]
        for i in range(6):

            if result1[i] == 1.:
                break
        prediction = classes[i]

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("template.html", image_name=filename, text=prediction)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run(debug=False)
