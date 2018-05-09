# Deep-Learning : Plate-recognition

Authors:
Gabriel Carestia
Timothée Clément
Nicolas Deffense
François Dusquesne
Thomas Feron
Antoine Rummens


-For French version, please see below-

Project goal: We used the Python environment - including Keras, Tensorflow, OpenCV, Tkinter libraries - to create and use a convolution neural network to read car plate characters.

Initially, we trained a convolutional neural network to recognize numbers and capital letters using the "emnist-balanced" database (86,400 training images, 14,400 test images). The images are black and white images of 28x28 pixels.
The'emnist-balanced' database is available via the following link: https://www.nist.gov/itl/iad/image-group/emnist-dataset
More information on the different EMNIST databases here: https://www.kaggle.com/crawford/emnist

Then we created a code that isolates the plate from the image of the entire car. Once this is done, the different characters are isolated (using the same method) and each character is read using the previously created neural network.

The "GUI-Voiture" code takes the image of an entire car while the "GUI-plaque" code takes the plate image directly.

Français:

But du projet: Nous avons utilisé l'environnement Python - dont les librairies Keras, Tensorflow, OpenCV, Tkinter - pour créer et utiliser un réseau de neurones à convolutions permettant de lire les caractères des plaques d'une voiture. 

Dans un premier temps, nous avons entraîné un réseau de neurons à convolution à reconnaître des chiffres et des lettres majuscules grâce à la base de données 'emnist-balanced' (86 400 images d'entraînement, 14 400 images de test). Les images sont des images noir et blancs 28x28 pixels.
La base de données 'emnist-balanced' est disponible via le lien suivant: https://www.nist.gov/itl/iad/image-group/emnist-dataset
Plus d'informations sur les différentes base de données EMNIST ici: https://www.kaggle.com/crawford/emnist

Ensuite, nous avons créé un code qui permet d'isoler la plaque de l'image de la voiture entière. Une fois cela fait, les différents caractères sont isolés (avec la même méthode) et chaque caractère est lu grâce au réseau de neurones créé auparavant.

Le code GUI-Voiture prend l'image d'une voiture entière tandis que le code GUI-plaque prend l'image de plaque directement.
