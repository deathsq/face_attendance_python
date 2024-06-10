import cv2
import face_recognition
import pickle
import os
# Import images
folderPath = 'Images'
pathList = os.listdir(folderPath)
# print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])
print(studentIds)

# find encodings

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         encode = face_recognition.face_encodings(img)[0]
         encodeList.append(encode)
    return encodeList

print("Encoding started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Completed")


file = open("EncodedFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File saved")
