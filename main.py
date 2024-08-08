import cv2
import os
import face_recognition


# Model architecture
prototxt = "model/deploy.prototxt"
# Weights
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

path = "Rostros"
facesEncodings = []
facesNames = []
for file_name in os.listdir(path):
     image = cv2.imread(path + "/" + file_name)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
     facesEncodings.append(f_coding)
     facesNames.append(file_name.split(".")[0])


image = cv2.imread("Reparto2.jpeg")
height, width, _ = image.shape
image_resized = cv2.resize(image, (300, 300))
# Create a blob
blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104, 117, 123))
print("blob.shape: ", blob.shape)
blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])

net.setInput(blob)
detections = net.forward()


for detection in detections[0][0]:
     # print("detection:", detection)
     if detection[2] > 0.5:

          box = detection[3:7] * [width, height, width, height]
          x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])

          img_rostro = image[y_start:y_end, x_start:x_end]
          img_rostro = cv2.cvtColor(img_rostro, cv2.COLOR_BGR2RGB)
          actual_face_encoding = face_recognition.face_encodings(img_rostro, known_face_locations=[(0, x_end-x_start, y_end-y_start, 0)])[0]
          result = face_recognition.compare_faces(facesEncodings, actual_face_encoding)
          # print(result)
          if True in result:
               index = result.index(True)
               name = facesNames[index]
               color = (125, 220, 0)
          else:
               name = "Desconocido"
               color = (50, 50, 255)

          cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, 2)
          cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, 2)
          cv2.putText(image, name, (x_start, y_start + y_end-y_start + 25), 2, 1, color, 2, cv2.LINE_AA)

cv2.imshow("Image", image)
# cv2.imshow("Rostro", img_rostro)
#cv2.imshow("blob_to_show", blob_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()