# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:56:07 2018

On va utiliser le jeu de données "balanced" parce qu'il n'y a pas une occurence plus forte
de retrouver certaines lettres dans des plaques de voiture.

+ d'infos sur le jeu de données ici: https://www.kaggle.com/crawford/emnist/version/3

@author: Timothée
"""
"""IMPORTER LES DONNEES (chiffres et lettres, images et labels) ET ENLEVER LES LETTRES MINUSCULES """
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from skimage.filters import threshold_otsu

data=sio.loadmat('emnist-balanced.mat')
data['dataset'].shape
data1=data['dataset']
data2=data1[0,0]

train_all=data2[0]
train_X_all=train_all['images']
train_X_all=train_X_all[0,0]
train_Y_all=train_all['labels']
train_Y_all=train_Y_all[0,0]

test_all=data2[1]
test_X_all=test_all['images']
test_X_all=test_X_all[0,0]
test_Y_all=test_all['labels']
test_Y_all=test_Y_all[0,0]


test_X=[]
test_Y=[]

for i in range(0,len(test_Y_all)):
    if 0<=test_Y_all[i]<=35:
        test_X.insert(i,test_X_all[i])
        test_Y.insert(i,test_Y_all[i])

test_X=np.array(test_X)
test_Y=np.array(test_Y)

train_X=[]
train_Y=[]

for i in range(0,len(train_Y_all)):
    if 0<=train_Y_all[i]<=35:
        train_X.insert(i,train_X_all[i])
        train_Y.insert(i,train_Y_all[i])

train_X=np.array(train_X)
train_Y=np.array(train_Y)


""" PREPROCESS DES DONNEES """
# reshape les vecteurs 784 en image 28x28, les rotationner et les retourner pour les voir bien
test_X2=np.zeros((len(test_X),28,28))
for i in range(0,len(test_X)):
    test_X2[i,:,:]=test_X[i].reshape(28,28)
    test_X2[i]=np.rot90(test_X2[i,:,:],3)
    test_X2[i]=np.fliplr(test_X2[i,:,:])
#    test_X2[i]=test_X2[i,4:25,4:25].resize((28,28))
    threshold_value = threshold_otsu(test_X2[i])
    test_X2[i]=test_X2[i]>threshold_value*1
    
train_X2=np.zeros((len(train_X),28,28))
for i in range(0,len(train_X)):
    train_X2[i,:,:]=train_X[i].reshape(28,28)
    train_X2[i]=np.rot90(train_X2[i],3)
    train_X2[i]=np.fliplr(train_X2[i,:,:])
#    train_X2[i]=train_X2[i,4:25,4:25].resize((28,28))
    threshold_value = threshold_otsu(train_X2[i])
    train_X2[i]=train_X2[i]>threshold_value*1
    
#on vérifie si on a bien ce qu'on voulait:
plt.figure(figsize=[5,5])
plt.subplot(121)
plt.imshow(train_X2[1458,:,:])
plt.title("Ground Truth:{}".format(train_Y[1458]))

plt.subplot(122)
plt.imshow(test_X2[989,:,:],cmap='gray')
plt.title("Ground Truth:{}".format(test_Y[989]))


train_X2 = train_X2.reshape(-1,28,28,1)
test_X2 = test_X2.reshape(-1,28,28,1)

train_X2 = train_X2.astype('float32')
test_X2 = test_X2.astype('float32')
#train_X2 = train_X2/255
#test_X2 = test_X2/255

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

#entrainement sur 80% de l'ensemble d'entrainement et validation sur 20% de l'ensemble d'entrainement:
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label=train_test_split(train_X2,train_Y_one_hot,test_size=0.2,random_state=13)
print('new training data shape, new validation data shape, new training label shape, new validation label shape:',train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

""" CONSTRUCTION DU CNN """
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D

#taille du batch, nombre d'itérations, et nombre de catégories:
batch_size=64
epochs=2
num_classes=36

#construction du réseau:
recon_plaque_model = Sequential()
recon_plaque_model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
recon_plaque_model.add(MaxPooling2D((2,2),padding='same'))
recon_plaque_model.add(Dropout(0.3))
recon_plaque_model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
recon_plaque_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
recon_plaque_model.add(Dropout(0.3))
recon_plaque_model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
recon_plaque_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
recon_plaque_model.add(Dropout(0.3))
recon_plaque_model.add(Flatten())
recon_plaque_model.add(Dense(128,activation='relu'))
recon_plaque_model.add(Dropout(0.3))
recon_plaque_model.add(Dense(num_classes,activation='softmax'))


#donner un résumé du modèle construit:
recon_plaque_model.summary()

#compiler le mode d'optimisation:
recon_plaque_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

""" Entraînement du CNN """
#entraîner le modèle:
recon_plaque_train = recon_plaque_model.fit(train_X,train_label,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X,valid_label))

#sauvegarder le modèle:
recon_plaque_model.save("recon_plaque_model.h5py")
""" Evaluation sur les données de test"""
test_eval=recon_plaque_model.evaluate(test_X2,test_Y_one_hot,verbose=0)
print('Test loss:',test_eval[0])
print('Test accuracy:',test_eval[1])

""" Visualisation de l'accuracy et du loss """
accuracy = recon_plaque_train.history['acc']
val_accuracy = recon_plaque_train.history['val_acc']
loss = recon_plaque_train.history['loss']
val_loss = recon_plaque_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

""" UTILISER LE CNN AVEC NOUVELLE IMAGE """
from PIL import Image
new_image = Image.open(r"C:\Users\Timothée\Documents\BIR\BIR21\Q2\questions spéciales de gestion de l'info\recon_plaque\new_image.png").convert('L')
new_image = new_image.resize((28,28))
new_image = np.array(new_image)
#visualiser l'image obtenue:
plt.figure(figsize=[5,5])
new_image = new_image.reshape(-1,28,28,1)
new_image = new_image.astype('float32')
#new_image = new_image/255
new_image = abs(new_image-1) #inverser l'image car ici j'ai une image noir sur blanc alors que les images d'entrainement c'est blanc sur noir
plt.imshow(new_image[0,:,:,0])
result = recon_plaque_model.predict(new_image)
print(result)
result = result.astype(float)
index_max_result = np.argmax(result)

if index_max_result==0:
    print("0")
elif index_max_result==1:
    print("1")
elif index_max_result==2:
    print("2")
elif index_max_result==3:
    print("3")    
elif index_max_result==4:
    print("4")
elif index_max_result==5:
    print("5")
elif index_max_result==6:
    print("6")
elif index_max_result==7:
    print("7")
elif index_max_result==8:
    print("8")
elif index_max_result==9:
    print("9")
if index_max_result==10:
    print("A")
elif index_max_result==11:
    print("B")
elif index_max_result==12:
    print("C")
elif index_max_result==13:
    print("D")    
elif index_max_result==14:
    print("E")
elif index_max_result==15:
    print("F")
elif index_max_result==16:
    print("G")
elif index_max_result==17:
    print("H")
elif index_max_result==18:
    print("I")
elif index_max_result==19:
    print("J")
elif index_max_result==20:
    print("K")
elif index_max_result==21:
    print("L")
elif index_max_result==22:
    print("M")
elif index_max_result==23:
    print("N")
elif index_max_result==24:
    print("O")
elif index_max_result==25:
    print("P")
elif index_max_result==26:
    print("Q")
elif index_max_result==27:
    print("R")
elif index_max_result==28:
    print("S")
elif index_max_result==29:
    print("T")
elif index_max_result==30:
    print("U")
elif index_max_result==31:
    print("V")
elif index_max_result==32:
    print("W")
elif index_max_result==33:
    print("X")
elif index_max_result==34:
    print("Y")
elif index_max_result==35:
    print("Z")

if result[0,index_max_result]<0.98:
    print("pas certain,c'est peut être un")
    tri=result.argsort()
    index_max_result=tri[0,34]
    if index_max_result==0:
        print("0")
    elif index_max_result==1:
        print("1")
    elif index_max_result==2:
        print("2")
    elif index_max_result==3:
        print("3")    
    elif index_max_result==4:
        print("4")
    elif index_max_result==5:
        print("5")
    elif index_max_result==6:
        print("6")
    elif index_max_result==7:
        print("7")
    elif index_max_result==8:
        print("8")
    elif index_max_result==9:
        print("9")
    if index_max_result==10:
        print("A")
    elif index_max_result==11:
        print("B")
    elif index_max_result==12:
        print("C")
    elif index_max_result==13:
        print("D")    
    elif index_max_result==14:
        print("E")
    elif index_max_result==15:
        print("F")
    elif index_max_result==16:
        print("G")
    elif index_max_result==17:
        print("H")
    elif index_max_result==18:
        print("I")
    elif index_max_result==19:
        print("J")
    elif index_max_result==20:
        print("K")
    elif index_max_result==21:
        print("L")
    elif index_max_result==22:
        print("M")
    elif index_max_result==23:
        print("N")
    elif index_max_result==24:
        print("O")
    elif index_max_result==25:
        print("P")
    elif index_max_result==26:
        print("Q")
    elif index_max_result==27:
        print("R")
    elif index_max_result==28:
        print("S")
    elif index_max_result==29:
        print("T")
    elif index_max_result==30:
        print("U")
    elif index_max_result==31:
        print("V")
    elif index_max_result==32:
        print("W")
    elif index_max_result==33:
        print("X")
    elif index_max_result==34:
        print("Y")
    elif index_max_result==35:
        print("Z")
    
    
from keras.models import load_model 
recon_plaque_model = load_model('recon_plaque_model.h5py')
result =recon_plaque_model.predict(new_image)



