import cv2
import face_recognition as fr
import pickle
import os
import matplotlib.pyplot as plt

roigen=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_roi(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    coords=roigen.detectMultiScale(gray,1.3,5)
    if len(coords)!=0:
        return coords
    else:
        return None
 
encodings=list()
labels=list()
labellist=os.listdir("/run/media/sangram/Games and Study Materials/Projects/Dlib fr/Data/")
for label in labellist:
    featurelist=os.listdir("/run/media/sangram/Games and Study Materials/Projects/Dlib fr/Data/"+label+"/")
    for feature in featurelist:
        img=cv2.cvtColor(cv2.imread("/run/media/sangram/Games and Study Materials/Projects/Dlib fr/Data/"+label+"/"+feature),cv2.COLOR_BGR2RGB)
        coords=get_roi(img)
        if coords is not None:
            for x,y,w,h in coords:
                img=img[y:y+w,x:x+h]
            
            plt.imshow(img)
            encoding=fr.api.face_encodings(img,known_face_locations=[(0,w,h,0)],num_jitters=5)
            encodings.append(encoding[0])
            labels.append(label)

data = {"encodings":encodings,"labels":labels}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()