# funcția de adăugare a zgomotului gaussian
def gaussian_noise(image_in,sigma):
    
    image_in=image_in.astype(np.float64) 
    
    l=image_in.shape[0];
    c=image_in.shape[1]
    
    zg=np.random.normal(0,sigma,[l,c]) # generarea numerelor aleatoare/zgomot aditiv gaussian
    
    im_out=image_in.copy()
    im_out[:,:,0]=image_in[:,:,0]+zg
    im_out[:,:,1]=image_in[:,:,1]+zg
    im_out[:,:,2]=image_in[:,:,2]+zg
    im_out=np.clip(im_out,0,255)
    im_out=im_out.astype(np.uint8)
    return im_out
# funcția pentru filtru de mediere marginală
def marginal_mean_filter(image_in):
    kernel=np.ones([3,3])/9 # kernel pentru filtru de mediere
    image_out=image_in.copy()

    for i in range(1,image_in.shape[0]-1):
        for j in range(1,image_in.shape[1]-1):
            for k in range(0,image_in.shape[2]):
                temp=image_in[i-1:i+2,j-1:j+2,k] 
                image_out[i,j,k]=np.floor(np.sum(temp*kernel)) # calcul pentru media aritmetica
    image_out=np.clip(image_out,0,255) 
    image_out=image_out.astype(np.uint8)
    return image_out
# calcul distanței euclediene ponderate
def weighted_eucledian_distance(patch1, patch2,kernel):
    return np.sum(np.sqrt((patch1-patch2)**2)*kernel) # diferența pixel cu pixel, înmulțit cu ponderi din kernelul gaussian

#funcția de estimare a unui pixel central în vecinătatea 7x7 și în centru ferestrei de căutare 21x21
def pixel_estimation(search_window,patch,kernel,h):
    
    window_height, window_width = search_window.shape # dimensiunea ferestrei de căutare
    patch_height, patch_width = patch.shape # dimensiunea vecinătății
    
    similarity = np.zeros((window_height-patch_height//2*2,window_width-patch_width//2*2))
    # matrice în care se stocează gradul de similaritate (distanță euclediană)
    
    #parcurgerea ferestrei de căutare
    for i in range(patch_height//2, window_height-patch_height//2):
        for j in range(patch_width//2, window_width-patch_width//2):
            #print(i,j)
            temp = search_window[i-patch_height//2:i+patch_height//2+1,j-patch_width//2:j+patch_width//2+1] #vecinătatea temporală, se parcurgere fiecare vecinătate posibilă în fereastra de 21x21
            similarity[i-patch_height//2,j-patch_width//2] = weighted_eucledian_distance(patch,temp,kernel) #calcul distanței euclediene ponderate
    
            #print(similarity[i-patch_height//2,j-patch_width//2])
    similarity = -similarity/(h*h) # aplicarea parametrului de filtrare h
    #print(similarity)
    
    Z = np.sum(np.exp(similarity)) # calcularea factorului de normalizare
    #print(Z)
    
    weights = np.zeros(search_window.shape) # inițializare matrice de ponderi
    
    for i in range(patch_height//2, window_height-patch_height//2):
        for j in range(patch_width//2, window_width-patch_width//2):
            weights[i,j] = 1/Z*np.exp(similarity[i-patch_height//2,j-patch_width//2]) # calculul ponderilor pentru fiecare pixel din fereastra de căutare
    
    #print(weights)
    #print(np.sum(weights))
    NLM_estimation = np.sum(weights*search_window) #ieșirea filtrului
    #print(NLM_estimation)
    return NLM_estimation.astype(np.uint8) # returnarea valorii pixelului
    #print(similarity.shape[0]*similarity.shape[1])

#definire filtru bazat pe algoritm NLM vecinătate 7x7, fereastră 21x21
def NLM_filter(image_in,kernel):
    image_out=image_in.copy()
    for i in range(10,image_in.shape[0]-10):
        for j in range(10,image_in.shape[1]-10):
            search_window=image_in[i-10:i+11,j-10:j+11]
            patch=image_in[i-3:i+4,j-3:j+4]
            image_out[i,j]=pixel_estimation(search_window,patch,kernel,100/256) #aplicarea estimării fiecărui pixel din imagine, cu excepția laturilor
    
    return image_out

#funcție pentru calculul erorii medii pătratice dintre două imagini
def mse(imageA, imageB):
    err = np.sum((imageA.astype(np.float) - imageB.astype(np.float)) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
    
import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

sigma = [5,10,20] # cele trei deviații standard folosite
file_names = ['baboon.bmp','goldhill.bmp','lena.png','Penguins.jpg','pepper.bmp'] #imagini folosite
cols = ['PSNR_noisy','MSE_noisy','PSNR_marginal','MSE_marginal','PSNR_NLM','MSE_NLM'] # numele coloanelor pentru creare de DataFrame care cuprinde informații privind măsuri de calitate

# creare de kernel gaussian 7x7
s, k = 1, 3 #  generare de kernel cu dimensiunea (2k+1)x(2k+1) medie=0 și sigma = s
probs = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)] 
kernel = np.outer(probs, probs)

#print (kernel)
#plt.imshow(kernel)
#plt.colorbar()
#plt.show()

metrics1 = np.zeros((5,6)) # matrice pentru stocarea măsurilor, ulterior transformat în DataFrame

for i,names in enumerate(file_names): # parcurgerea celor 5 imagini pentru experiment1
    image=io.imread(names)
    plt.figure(figsize=(5,5))
    plt.imshow(image)

    image_noise=gaussian_noise(image,sigma[1]) # adăugarea zgomot gaussian de deviație standard 10
    
    metrics1[i,0] = cv2.PSNR(image,image_noise) # PSNR imagine originală - imagine zgomotos
    metrics1[i,1] = mse(image,image_noise) # MSE imagine originală - imagine zgomotos

    plt.figure(figsize=(5,5))
    plt.imshow(image_noise)
          
    image_filtered_marginal = marginal_mean_filter(image_noise) # filtrare marginală
    
    metrics1[i,2] = cv2.PSNR(image,image_filtered_marginal)  # PSNR imagine originală - imagine filtrată cu filtru de mediere marginală
    metrics1[i,3] = mse(image,image_filtered_marginal) # MSE imagine originală - imagine filtrată cu filtru de mediere marginală
    
    #separarea celor 3 canale R,G,B al imaginii zgomotoase
    image_R = image_noise[:,:,0]
    image_G = image_noise[:,:,1]
    image_B = image_noise[:,:,2]

    #aplicarea filtru bazat pe algoritm NLM pentru fiecare canal individual
    image_filtered_NLM[:,:,0] = NLM_filter(image_R,kernel)
    image_filtered_NLM[:,:,1] = NLM_filter(image_G,kernel)
    image_filtered_NLM[:,:,2] = NLM_filter(image_B,kernel)
    
    metrics1[i,4] = cv2.PSNR(image,image_filtered_NLM) # PSNR imagine originală - imagine filtrată cu filtru de mediere marginală
    metrics1[i,5] = MSE(image,image_filtered_NLM) # MSE imagine originală - imagine filtrată cu filtru de mediere marginală

metrics_sigma10 = pd.DataFrame(metrics1,file_names,cols)  #generarea de DataFrame pentru vizualizare corespunzătoare, eventual generare de fișier .csv  
print('Metrici de calitate pentru experimentul 1')
print(metrics_sigma10)
   
rows = ['baboon5','lena5','baboon10','lena10','baboon20','lena20'] # numele rândurilor din DataFreame metrics2
cols = ['PSNR_noisy','MSE_noisy','PSNR_marginal','MSE_marginal','PSNR_NLM','MSE_NLM'] #numele coloanelor din DataFrame metrics2
metircs2 = np.zeros((6,6)) #matrice pentru stocare

for i,std in enumerate(sigma):#parcugerea celor 3 intensități de zgomot
    #citirea imaginilor 
    image1=io.imread('lena.png')
    image2=io.imread('baboon.bmp')
    
    #adăugare de zgomot gaussian
    image_noise1=gaussian_noise(image,std)
    image_noise2=gaussian_noise(image,std)
    
    # calcul PSNR, MSE dintre imaginile originale și cele zgomotoase
    metrics2[2*i,0] = cv2.PSNR(image1,image_noise1)
    metrics2[2*i,1] = mse(image1,image_noise1)
    metrics2[2*i+1,0] = cv2.PSNR(image2,image_noise2)
    metrics2[2*i+1,1] = mse(image2,image_noise2)
    
    # aplicarea filtrului de mediere marginală
    image_filtered_marginal1 = marginal_mean_filter(image_noise1)
    image_filtered_marginal2 = marginal_mean_filter(image_noise2)

    #calcul PSNR, MSE dintre imaginea originală și cele filtrate cu filtru marginal
    metrics2[2*i,2] = cv2.PSNR(image1,image_filtered_marginal1)
    metrics2[2*i,3] = mse(image1,image_filtered_marginal1)
    metrics2[2*i+1,2] = cv2.PSNR(image2,image_filtered_marginal2)
    metrics2[2*i+1,3] = mse(image2,image_filtered_marginal2)
    
    #aplicarea algoritmului NLM pentru fiecare canal în parte
    image_R1 = image_noise1[:,:,0]
    image_G1 = image_noise1[:,:,1]
    image_B1 = image_noise1[:,:,2]

    image_filtered_NLM1[:,:,0] = NLM_filter(image_R1,kernel)
    image_filtered_NLM1[:,:,1] = NLM_filter(image_G1,kernel)
    image_filtered_NLM1[:,:,2] = NLM_filter(image_B1,kernel)
    
    image_R2 = image_noise2[:,:,0]
    image_G2 = image_noise2[:,:,1]
    image_B2 = image_noise2[:,:,2]

    image_filtered_NLM2[:,:,0] = NLM_filter(image_R2,kernel)
    image_filtered_NLM2[:,:,1] = NLM_filter(image_G2,kernel)
    image_filtered_NLM2[:,:,2] = NLM_filter(image_B2,kernel)

    #calcul PSNR, MSE dintre imaginea originală și cele filtrate cu filtru bazat pe Non-Local Means
    metrics2[2*i,4] = cv2.PSNR(image1,image_filtered_NLM1)
    metrics2[2*i,5] = mse(image1,image_filtered_NLM1)
    metrics2[2*i+1,4] = cv2.PSNR(image2,image_filtered_NLM2)
    metrics2[2*i+1,5] = mse(image2,image_filtered_NLM2)

metrics_variable_sigma= pd.DataFrame(metrics2,rows,cols)  # Generare de DataFrame pentru măsuri de calități al experimentului 2
print('Metrici de calitate pentru experimentul 2')
print(metrics_variable_sigma)