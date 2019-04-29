import cv2
import face_recognition as fr
import numpy as np
import pickle

roigen=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_roi(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    coor=roigen.detectMultiScale(gray,1.3,5)
    if len(coor)!=0:
        return coor
    else:
        return None
    
def get_dict(truthvals,labels):
    face_matches=dict()
    truthvals=np.array(truthvals)
    labels=labels[truthvals]
    for i in labels:
        if i in face_matches:
            face_matches[i]+=1
        else:
            face_matches[i]=1
    return face_matches
 
    
file=open("encodings.pickle","rb")
data=pickle.loads(file.read())
file.close()

labels=data["labels"]
labels=np.array(labels)
encodings=data["encodings"]

cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=cv2.flip(frame,1)
    coords=get_roi(frame)
    if coords is not None:
        for (x,y,w,h) in coords:
            roi=frame[y:y+w,x:x+h]
            current_enc=fr.face_encodings(roi,known_face_locations=[(0,w,h,0)],num_jitters=1)
            frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if len(current_enc)>0:
                truthvals=fr.compare_faces(encodings,current_enc[0],tolerance=0.6)
                matches=get_dict(truthvals,labels)
                if len(matches)==0:
                    name="Unknown"
                else:
                    name=max(matches)
                cv2.putText(frame,name,(x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                
    cv2.imshow("frame",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()