import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from PIL import Image

print("Hi! Hello there! Greetings of the day! Welcome to REG! Race-Emotion-Gender Identifier!")

while(True):

    choice = int(input("Please pick an image to test from the following:- \n 1) Steve Jobs \n 2) JK Rowling \n 3) Will Smith \n 4) Oprah Winfrey \n 5) Sundar Pichai \n 6) Maitreyi Ramakrishnan \n 7) Kim Jong Un \n 8) Shen Xiaoting \n Enter your choice with the number assigned to the person! \n Choice: "))

    choice = str(choice)
    img = cv2.imread("people/" + choice + ".jpg")

    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    prediction = DeepFace.analyze(color_img)

    #Face detection and drawing rectangle

    #loading our xml file into faceCascade using cv2.CascadeClassifier
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    #detecting face in color_image and getting 4 points(x,y,u,v) around face from the image, and assigning those values to 'faces' variable 
    faces = faceCascade.detectMultiScale(color_img, 1.1, 4)

    #Showing emotion, gender & race

    #choose font for text
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, u, v) in faces:
        # draw the predicted face name on the image
        cv2.rectangle(color_img, (x,y), (x+u, y+v), (0, 255, 2), 2)
        
        cv2.putText(color_img, "Gender: " + prediction['gender'], (x, y+v + 25), font, 1, (0, 255, 0), 2)
        cv2.putText(color_img, "Emotion: " + prediction['dominant_emotion'], (x, y+v + 50), font, 1, (0, 255, 0), 2)
        cv2.putText(color_img, "Race: " + prediction['dominant_race'], (x, y+v + 75), font, 1, (0, 255, 0), 2)

    #finally displaying image
    image = Image.fromarray(color_img)
    image.show()

    endchoice = int(input("Do you want to try another image? \n If yes, enter 1! \n If No, enter 2! \n Choice: "))
    if(endchoice == 1):
        pass

    if(endchoice == 2):
        print("\n The Good Greetings of The Bye!")
        exit()