# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:11:26 2018

@author: Timothée
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.io import imread
from skimage.filters import threshold_otsu, threshold_yen
from skimage import measure
from skimage.measure import regionprops


#############################################################
############### IMPORTATION ET PRE-TRAITEMENT ###############
#############################################################


car_image = imread("plaque.jpg", as_grey=True)
gray_car_image = car_image * 255
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value*1


#############################################################
### SELECTION DE LA REGION DE L'IMAGE CONTENANT LA PLAQUE ### 
#############################################################

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray");
label_image = measure.label(binary_car_image)
plate_dimensions = (0.05*label_image.shape[0], 0.8*label_image.shape[0], 0.05*label_image.shape[1], 0.8*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []


for region in regionprops(label_image):
    if region.area < 50:
        continue
    
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
        
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        if region_width/region_height>3 and region_width/region_height<8:
            plate_like_objects.append(binary_car_image[min_row:max_row,min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)


plt.show()

index=[]

if len(plate_objects_cordinates)==1:
    index=0
else:        
    count=np.zeros([len(plate_objects_cordinates),2])
    for k in range(0,len(plate_objects_cordinates)):
    
        plate =car_image[plate_objects_cordinates[k][0]:plate_objects_cordinates[k][2],plate_objects_cordinates[k][1]:plate_objects_cordinates[k][3]]
        plt.imshow(plate)
        threshold_value = threshold_otsu(plate)
        binary_car_image = plate > threshold_value
        plate=binary_car_image*1
    
        a=round(1/3*plate.shape[0])
        for i in range(1,plate.shape[1]):
            if plate[a][i]!=plate[a][i-1]:
                count[k][0]+=1
        
        count[k][0]=  count[k][0]/2
    
        b=round(2/3*plate.shape[0])
        for i in range(1,plate.shape[1]):
            if plate[b][i]!=plate[b][i-1]:
                count[k][1]+=1
        
        count[k][1]= count[k][1]/2
    count_tot=np.zeros([len(plate_objects_cordinates),1])
    for k in range(0, len(plate_objects_cordinates)):
        count_tot[k]=(count[k][0]+count[k][1])/2
        
    for k in range(len(plate_objects_cordinates)):
        if count_tot[k][0]>6 and count_tot[k][0]<16:
            index=k
        
if index==[]:
    print('la plaque n a pas été trouvé dans l image : 2e essaie avec un autre type de traitement d image')
Good_boy =car_image[plate_objects_cordinates[index][0]:plate_objects_cordinates[index][2],plate_objects_cordinates[index][1]:plate_objects_cordinates[index][3]]
plt.imshow(Good_boy)


####################################################################
### SELECTION DE LA REGION DE LA PLAQUE CONTENANT DES CARACTERES ### 
####################################################################


threshold_value = threshold_otsu(Good_boy)
license_plate = Good_boy > threshold_value
license_plate = np.invert(license_plate)

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

character_dimensions = (0.3*license_plate.shape[0], 0.90*license_plate.shape[0], 0.01*license_plate.shape[1], 0.2*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
counter=0
column_list = []
char_like_objects= []
char_objects_cordinates = []
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:

        char_like_objects.append(license_plate[y0:y1, x0:x1])
        char_objects_cordinates.append((y0, x0,y1,x1))
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        
ax1.set_axis_off()
plt.tight_layout()
plt.show()

char_ord=sorted(char_objects_cordinates,key=lambda char_objects_cordinates: char_objects_cordinates[1])

a=[]
for i in range (0,len(char_ord)):
    a.append(license_plate[char_ord[i][0]:char_ord[i][2],char_ord[i][1]:char_ord[i][3]])
    
border=[]
for t in range (len(a)):
    a[t]=a[t]*255
    cv2.imwrite('Character.png',a[t])
    im=cv2.imread('Character.png')
    bordersize = round(0.2*max(im.shape))
    border.append(cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0]))


    



from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model 
recon_plaque_model = load_model('recon_plaque_model.h5py')
from tkinter import Tk, Label, Button

border2=[]
for i in range (len(border)):
    border2.append(border[i][:,:,0])
        
r=0 #row
c=1 #column
compt=-1

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("Reconnaissance de plaque")

        self.label = Label(master, text="Veuillez copier dans le dossier 'C:/Users/Timothée/Documents/BIR/BIR21/Q2/questions spéciales de gestion de l'info/recon_plaque' l'image de votre plaque, sous le nom 'plaque.jpg'").grid(row=0,column=1)

        self.run_button = Button(master, text="Exact, reconnaître le caractère suivant", command=self.run).grid(row=1,column=0)
        
        self.faux_button = Button(master,text="Faux, autre possibilité", command=self.faux).grid(row=2,column=0)
        

    def run(self):
        global r
        r=0
        global c
        c=c+1
        global compt
        compt=compt+1
        b=border2[compt]
        b=b.astype(np.int32)
        new_image=Image.fromarray(b)
        new_image = new_image.resize((28,28))
        new_image = np.array(new_image)
        #visualiser l'image obtenue:
        plt.figure(figsize=[5,5])
        new_image = new_image.reshape(-1,28,28,1)
        new_image = new_image.astype('float32')
        new_image = new_image/255
        #        new_image = abs(new_image-1) #inverser l'image car ici j'ai une image noir sur blanc alors que les images d'entrainement c'est blanc sur noir
        plt.imshow(new_image[0,:,:,0])
        global result
        result = recon_plaque_model.predict(new_image)
        print(result)
        result = result.astype(float)
        index_max_result = np.argmax(result)
        if 0<=index_max_result<=9:
            resultat=index_max_result
            print("index_max_result")
            if compt<=2 and index_max_result==7:
                resultat=1
        if index_max_result==10:
                print("A")
                resultat='A'
        elif index_max_result==11:
                print("B")
                resultat='B'
        elif index_max_result==12:
                print("C")
                resultat='C'
        elif index_max_result==13:
                print("D")
                resultat='D'
        elif index_max_result==14:
                print("E")
                resultat='E'
        elif index_max_result==15:
                print("F")
                resultat='F'
        elif index_max_result==16:
                print("G")
                resultat='G'
        elif index_max_result==17:
                print("H")
                resultat='H'
        elif index_max_result==18:
                print("I")
                resultat='I'
        elif index_max_result==19:
                print("J")
                resultat='J'
        elif index_max_result==20:
                print("K")
                resultat='K'
        elif index_max_result==21:
                print("L")
                resultat='L'
        elif index_max_result==22:
                print("M")
                resultat='M'
        elif index_max_result==23:
                print("N")
                resultat='N'
        elif index_max_result==24:
                print("O")
                resultat='O'
        elif index_max_result==25:
                print("P")
                resultat='P'
        elif index_max_result==26:
                print("Q")
                resultat='Q'
        elif index_max_result==27:
                print("R")
                resultat='R'
        elif index_max_result==28:
                print("S")
                resultat='S'
        elif index_max_result==29:
                print("T")
                resultat='T'
        elif index_max_result==30:
                print("U")
                resultat='U'
        elif index_max_result==31:
                print("V")
                resultat='V'
        elif index_max_result==32:
                print("W")
                resultat='W'
        elif index_max_result==33:
                print("X")
                resultat='X'
        elif index_max_result==34:
                print("Y")
                resultat='Y'
        elif index_max_result==35:
                print("Z")
                resultat='Z'
        label = Label(root, text= str(resultat)).grid(row=3,column=c)
            
    def faux(self):
            print('Alorsc''est peut être un')
            global r
            r=r+1
            global tri
            tri=result.argsort()
            index_max_result=tri[0,35-r]
            
            if 0<=index_max_result<=9:
                resultat=index_max_result
                print("index_max_result")
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
                resultat='A'
            elif index_max_result==11:
                print("B")
                resultat='B'
            elif index_max_result==12:
                print("C")
                resultat='C'
            elif index_max_result==13:
                print("D")
                resultat='D'
            elif index_max_result==14:
                print("E")
                resultat='E'
            elif index_max_result==15:
                print("F")
                resultat='F'
            elif index_max_result==16:
                print("G")
                resultat='G'
            elif index_max_result==17:
                print("H")
                resultat='H'
            elif index_max_result==18:
                print("I")
                resultat='I'
            elif index_max_result==19:
                print("J")
                resultat='J'
            elif index_max_result==20:
                print("K")
                resultat='K'
            elif index_max_result==21:
                print("L")
                resultat='L'
            elif index_max_result==22:
                print("M")
                resultat='M'
            elif index_max_result==23:
                print("N")
                resultat='N'
            elif index_max_result==24:
                print("O")
                resultat='O'
            elif index_max_result==25:
                print("P")
                resultat='P'
            elif index_max_result==26:
                print("Q")
                resultat='Q'
            elif index_max_result==27:
                print("R")
                resultat='R'
            elif index_max_result==28:
                print("S")
                resultat='S'
            elif index_max_result==29:
                print("T")
                resultat='T'
            elif index_max_result==30:
                print("U")
                resultat='U'
            elif index_max_result==31:
                print("V")
                resultat='V'
            elif index_max_result==32:
                print("W")
                resultat='W'
            elif index_max_result==33:
                print("X")
                resultat='X'
            elif index_max_result==34:
                print("Y")
                resultat='Y'
            elif index_max_result==35:
                print("Z")
                resultat='Z'
            label = Label(root, text= str(resultat)).grid(row=3+r,column=c)
           
root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()
