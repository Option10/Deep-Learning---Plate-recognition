from PIL import Image
import cv2
import numpy as np
from numpy import*
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
from skimage import data
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize 
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border


im=Image.open("plaque.jpg")
#im=im.rotate(1)
im.save("imma_12.png")
im2=im.convert("L")
im2.save("imma_22.png")
 

threshold = 100
im = im2.point(lambda p: p > threshold and 255)
im.save("imma_32.png")
img="imma_32.png"

basewidth = 400
img = Image.open("imma_32.png")
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save("imma_42.png")

#-----------------------------------------------------
temp=Image.open('imma_42.png')
temp=temp.convert('1')      # Convert to black&white
A = array(temp)             # Creates an array, white pixels==True and black pixels==False
new_A=empty((A.shape[0],A.shape[1]),None)    #New array with same size as A

for i in range(len(A)):
    for j in range(len(A[i])):
        if A[i][j]==True:
            new_A[i][j]=0
        else:
            new_A[i][j]=1
#print(new_A)

#elevation_map = sobel(new_A)
#fig, ax = plt.subplots(figsize=(4, 3))
#ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
#ax.set_title('elevation map')
#ax.axis('off')

# label image regions
label_image = label(new_A)
image_label_overlay = label2rgb(label_image, image=new_A)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.imshow(image_label_overlay)

char_objects_cordinates = []
char_like_objects = []

for region in regionprops(label_image):
    # take regions with large enough areas
    # skip small images
    if region['Area'] < 135:
        continue

    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    
    if region['Area'] >= 135:
        char_like_objects.append(A[min_row:max_row,min_col:max_col])
        char_objects_cordinates.append((min_row, min_col,max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="green", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

        
ax1.set_axis_off()
plt.tight_layout()
plt.show()

number_of_candidates = len(char_like_objects)


for i in range(number_of_candidates):

        prop= np.invert(char_like_objects[i])
        bw_prop = prop*255
        labelled_prop = measure.label(bw_prop)
        #On affiche la plaque
        fig, ax1 = plt.subplots()
        ax1.imshow(bw_prop, cmap="gray")
        plt.show()


#regionprops(label_image,intensity_image=True)





