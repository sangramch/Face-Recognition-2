import cv2
import face_recognition as fr
import pickle
import time

roigen=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_roi(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    coor=roigen.detectMultiScale(gray,1.3,5)
    if len(coor)!=0:
        return coor
    else:
        return None
 
file=open("encodings.pickle","rb")
data=pickle.loads(file.read())
file.close()

encodings=data["encodings"]
labels=data["labels"]

newlabel=input("Enter Name: ")

while True:
    cap=cv2.VideoCapture(0)
    time1=time.time()
    while True:
        _,frame=cap.read()
        cv2.imshow("frame",frame)
        if (cv2.waitKey(1) & (time.time()-time1>5)):
            break
    
    _,frame=cap.read()
    cap.release()
    cv2.destroyWindow("frame")
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    coord=get_roi(frame)
    
    if coord is not None:
        for (x,y,w,h) in coord:
            roi=frame[y:y+w,x:x+w]
            current_enc=fr.face_encodings(roi,known_face_locations=[(0,w,h,0)],num_jitters=5)
            frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("frame",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            encodings.append(current_enc[0])
            labels.append(newlabel)
    else:
        print("No faces found")
        cv2.imshow("frame2",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ch=input("Add again? (Y/n): ")
    if ch=="y" or ch=="Y":
        continue
    else:
        break

data = {"encodings":encodings,"labels":labels}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()