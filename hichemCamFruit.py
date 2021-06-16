# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import activations
# from tensorflow.python.keras import backend
# from tensorflow.python.keras.utils import data_utils
# from tensorflow.python.util.tf_export import keras_export
#
# # je vais ici tenter de reconnaitre le fruit avec opencv
# # exemple d'usage : @keras_export('keras.applications.imagenet_utils.preprocess_input')
#
# image = cv2.imread(file)
# image = image.transpose((2, 0, 1))
#
# origin = cv2.imread(file)
# cv2.putText(origin, "Predict: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
# cv2.imshow("Result", origin)


##########################################################
import tkinter as tk
import cv2
import tensorflow as tf
import openpyxl as xl

from openpyxl.chart import BarChart, Reference
from openpyxl import Workbook
from tkinter import ttk
from typing import Container
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# ces classes fonctionnes avec le model13.h5
# classes = ['Apple Braeburn', 'Apple Golden 1', 'Blueberry', 'Cherry 1', 'Cherry 2', 'Fresh Banana', 'Fresh Orange',
#            'Huckleberry', 'Litchi', 'Maracuja', 'Rotten Banana', 'Rotten Orange']
# oufff enfin j'ai fini de faire les classe jen ai chier a les mettre en ordre et savoir
# le nombre et ce qui manque dans la liste des classe en dur par rapport au dossier  hahaha'
# classes au nombre de 65 fonctionne avec le model65.h5
# classes = ['Apple Braeburn', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith',
#            'Apple Red Delicious', 'Apple Red Yellow', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
#            'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Red',
#            'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cherry Rainier',
#            'Cherry 1', 'Cherry 2', 'Clementine', 'Cocos', 'Dates',
#            'Granadilla', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grapefruit Pink',
#            'Grapefruit White', 'Guava', 'Huckleberry', 'Kaki', 'Kiwi',
#            'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Litchi',
#            'Mandarine', 'Mango', 'Maracuja', 'Melon Piel de Sapo', 'Nectarine',
#            'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat',
#            'Pear', 'Pear Abate', 'Pear Monster', 'Pear Williams', 'Pepino',
#            'Pineapple', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Quince',
#            'Raspberry', 'Salak', 'Strawberry', 'Tamarillo', 'Tangelo']


# 131 fruit
classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
           'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2',
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
           'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White',
           'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat', 'Peach 2', 'Pear', 'Pear Abate',
           'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pear 2', 'Pepino',
           'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk',
           'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2',  'Plum 3', 'Pomegranate', 'Pomelo Sweetie',
           'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry',
           'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato Cherry Red',
           'Tomato Heart', 'Tomato Maroon', 'Tomato Not Ripened', 'Tomato Yellow', 'Tomato 1', 'Tomato 2',
           'Tomato 3', 'Tomato 4', 'Walnut', 'Watermelon']

# # pour reconnaitre uniquement 6 fruit j'ai mis ça en place,
# cela fonctionne plutot bien avec les deux appli de prediction, le model pour ces 5 fruit est : model5.h5
# classes = ['apple', 'banana', 'orange', 'litchi', 'maracuja']


def show_frame(frame):
    frame.tkraise()


def createwidgets(frame):
    root.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)
    show_frame(frame1)

    frame.upload = Button(frame, text="Predict Uploaded", bg="#B5EAD7", command=lambda: FruitRec(openDirectory),
                          font=('Courier New', 15), width=16)
    frame.upload.grid(row=4, column=4)

    frame.predict = Button(frame, text="Predict Captured", bg="#B5EAD7", command=lambda: FruitRec(imgName),
                           font=('Courier New', 15), width=16)
    frame.predict.grid(row=4, column=5)

    frame.cameraLabel = Label(frame, bg="steelblue", borderwidth=3, relief="groove")
    frame.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)

    frame.captureBTN = Button(frame, text="CAPTURE", command=Capture, bg="#B5EAD7", font=('Courier New', 15), width=20)
    frame.captureBTN.grid(row=4, column=1, padx=10, pady=10)

    frame.CAMBTN = Button(frame, text="STOP CAMERA", command=StopCAM, bg="#B5EAD7", font=('Courier New', 15), width=13)
    frame.CAMBTN.grid(row=4, column=2)

    frame.previewlabel = Label(frame, fg="black", text=" quelle est ce fruit !", font=('Courier New', 20))
    frame.previewlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    frame.imageLabel = Label(frame, bg="steelblue", borderwidth=3, relief="groove")
    frame.imageLabel.grid(row=2, column=4, padx=10, pady=10, columnspan=2)

    frame.openImageEntry = Entry(frame, width=55, textvariable=imagePath)
    frame.openImageEntry.grid(row=3, column=4, padx=10, pady=10)

    frame.openImageButton = Button(frame, width=10, text="BROWSE", command=imageBrowse)
    frame.openImageButton.grid(row=3, column=5, padx=10, pady=10)

    ShowFeed()


def ShowFeed():
    ret, frame = root.cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (0, 255, 255))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        videoImg = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=videoImg)
        frame1.cameraLabel.configure(image=imgtk)
        frame1.cameraLabel.imgtk = imgtk
        frame1.cameraLabel.after(10, ShowFeed)
    else:
        frame1.cameraLabel.configure(image='')


def imageBrowse():
    global imageView
    global openDirectory
    openDirectory = filedialog.askopenfilename(initialdir="HichemFruitClassification")
    imagePath.set(openDirectory)
    imageView = Image.open(openDirectory)
    imageResize = imageView.resize((640, 480), Image.ANTIALIAS)
    imageDisplay = ImageTk.PhotoImage(imageResize)
    frame1.imageLabel.config(image=imageDisplay)
    frame1.imageLabel.photo = imageDisplay


def Capture():
    global imgName
    image_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S')
    image_path = r'SavedPics'
    imgName = image_path + '/' + image_name + ".jpg"
    ret, frame = root.cap.read()
    cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (430, 460), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (0, 255, 255))
    success = cv2.imwrite(imgName, frame)
    saved_image = Image.open(imgName)
    saved_image_flip = saved_image.transpose(Image.FLIP_LEFT_RIGHT)
    saved_image = ImageTk.PhotoImage(saved_image_flip)
    frame1.imageLabel.config(image=saved_image)

    frame1.imageLabel.photo = saved_image


def StopCAM():
    root.cap.release()

    frame1.CAMBTN.config(text="START CAMERA", command=StartCAM)

    frame1.cameraLabel.config(text="OFF CAM", font=('Courier New', 70))


def StartCAM():
    root.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    frame1.CAMBTN.config(text="STOP CAMERA", command=StopCAM)

    frame1.cameraLabel.config(text="")

    ShowFeed()


def FruitRec(prev_image):
    root.cap.release()
    show_frame(frame2)

    frame2.backButton = Button(frame2, width=10, text="Back", command=lambda: createwidgets(frame1))
    frame2.backButton.grid(row=1, column=0, padx=10, pady=10)

    frame2.imageLabel = Label(frame2, bg="steelblue", borderwidth=3, relief="groove")
    frame2.imageLabel.grid(row=3, column=1, padx=10, pady=10, columnspan=2)

    Upload(prev_image)


def Upload(prev_image):
    saved_image = Image.open(prev_image)
    saved_image = saved_image.resize((640, 480), Image.ANTIALIAS)

    saved_image_flip = saved_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Creating object of PhotoImage() class to display the frame
    saved_image = ImageTk.PhotoImage(saved_image_flip)
    # Configuring the label to display the frame
    frame2.imageLabel.config(image=saved_image)
    # Keeping a reference
    frame2.imageLabel.photo = saved_image

    Predict(prev_image)


def Predict(prev_image):
    global prediction
    # fonctionne avec les classes au nombre de 65
    # new_model = load_model('model65.h5')
    # les autres models sont entrainer a reconnaitre 131 fruit "classe" mais ils me pose probleme
    # new_model = load_model('modelInception.h5')
    # new_model = load_model('inceptionDernier.h5')
    # ce model fonctionne quelques fois avec l'upload mais avec la camera direct
    new_model = load_model('dernierModelCnnNewMoreFruit.h5')
    # model plutot bon avec les 5 fruit ,
    # oublie pas de commenté les classes et de decommenté les classes contenant que les 5 fruit
    # new_model = load_model('model5.h5')
    # model avec 13 fruit decommenté la ligne 40
    # new_model = load_model('model13.h5')
    new_model.summary()
    test_image = image.load_img(
        prev_image,
        # target_size=(64, 64))
        # a utilisé pour les model venu de cnn-new-more-fruit
        target_size=(100, 100))
        # à utilisé quand c'est le model inception target 299, 299
        # target_size=(299, 299))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)
    result1 = result[0]
    for i in range(36):

        if result1[i] == 1.:
            break
    prediction = classes[i]

    def excel():
        list.append(prediction)
        sheet.append([prediction])
        wb.save('Records.xlsx')
        print(list)
        messagebox.showinfo("SUCCESS", f'''INFORMATION RECORDED.
PRESS TRY AGAIN TO CAPTURE ANOTHER''')

    food = f'''
    Fruit :{prediction}
 Force :ere
    Prix: erere  '''

    frame2.previewlabel = Label(frame2, fg="black", text=food, font=('Courier New', 15))
    frame2.previewlabel.grid(row=3, column=6, padx=10, pady=10, columnspan=3)

    frame2.cancel = Button(frame2, text="Try Again", bg="#B5EAD7", command=lambda: createwidgets(frame1),
                           font=('Courier New', 15), width=10)
    frame2.cancel.grid(row=4, column=6)

    frame2.save = Button(frame2, text="Save", bg="#B5EAD7", command=excel,
                         font=('Courier New', 15), width=10)
    frame2.save.grid(row=4, column=7)


root = tk.Tk()
root.state('zoomed')
root.title('Seriket-Hichem')

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

destPath = StringVar()
imagePath = StringVar()

global list
global sheet
list = []

wb = xl.load_workbook('Records.xlsx')
sheet = wb['Sheet']

frame1 = tk.Frame(root)
frame2 = tk.Frame(root)
frame3 = tk.Frame(root)

for frame in (frame1, frame2, frame3):
    frame.grid(row=0, column=0, sticky='nsew')

createwidgets(frame1)
root.mainloop()
