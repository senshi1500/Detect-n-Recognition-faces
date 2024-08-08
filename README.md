
# Detect Face reconiction with python

This is a simple project for detection and recnition face 
The scrip worck whit an pretrainer model named "res10_300x300_ssd_iter_140000" It use that model to find the faces in an image when find out, use the library reconiction face, This library allows you to compare two faces and define who is most similar.

In this example I take an image of Malcom's family in the middle, I use the res10 model to obtain the faces to detect and I save them in the folder where I save the faces to detect,
Then I take another image and repeat the process of searching for faces and when it finds them it compares them with the previously saved faces and when it finds them it draws a box over the face with the name of the character.

Special thanks OMES for this tutorial for the detection of faces: https://omes-va.com/deteccion-facial-dnn-opencv-python/