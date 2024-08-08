import cv2



# Model architecture
prototxt = "model/deploy.prototxt"
# Weights
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)


image = cv2.imread("RepartoMalcomElDeEnMedio.jpg")
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
          # i=i+1
          box = detection[3:7] * [width, height, width, height]
          x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
          # Guradamos los rostros de una unica imagen para que sean reconocidos
          # img_rostro = image[y_start:y_end, x_start:x_end]
          # cv2.imwrite(f"Rostros/img_rostro{i}.jpg", img_rostro)
          cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
          cv2.putText(image, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)

cv2.imshow("Image", image)
# cv2.imshow("Rostro", img_rostro)
#cv2.imshow("blob_to_show", blob_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()