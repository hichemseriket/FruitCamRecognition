ici je doit deja expliqué les installation necessaire
l'environnement conda tensorflow et les dependance et commenet jai créé mon environement python pour tensorflow gpu
cnn.py : jai fait un script pour entrainer et sortir un model capable de differencie sur 3 fruit les fresh et les pourri, ca me sort le model : model.h5
app.py : dedans un serveur web pour testé la reconnaissance dde fruit selon le model entrainné dans le cnn.py
cnn-new-more-fruit : un autre reseau de nouronne capable d'entrainer un model a reconnaitre plus de fruit ici jai mis 65 classes
le dossier reconnaissance d objet contient du code pour activer la camera avec reconnaissance grace a un model preentrainé je compte men servir pour faire fct la cam avec les fruit
le detection fonctonne bien avec le model ssd du net
sauf que il utilise un frozen graph en plus donc mon model seule suffit pas a lancé le truc a voir

quand je rajoute des data entrainement et test il faut changé le path si besoin , les classes dans le app ou dans le script qui lance la video et faut surtout changé le nombre de neuronne en sortie
da  ns la fonction qui ajoute les dense faut changé le parame unit dans la ligne ou on fait le softmax au nombre de sortie qui est le nombre de classes


telechargement des data avec a ce jour 131 classe : https://www.kaggle.com/moltean/fruits

cnn.py : script CNN pour entrainer et sortir un model, faut adapter les routes et le nombre de classes pour correspondre au nombre de classes dans les dataset

app.py : script serveur flask pour tester le model sorti

cnnMoreFrruit360.py : new script avec model_inception_v3, ce reseau de neuronne est plus performant que le premier, j'ai juste un souci avec la sauvgarde du model , du coup j'ai integré le serveur flask dedans pour pouvoir utlisé le model in script directement

hichemCamFruit.py : script pour utlisé le model avec une appli opencv et pouvoir reconnaitre directement via camera le fruit, pour le moment cela me sert  a prendre une photo et la predire , donc pas live recgnition pour le moment.