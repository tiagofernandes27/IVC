import cv2 as cv2
import numpy as np

def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)
    # rest of init


def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)
    ret, image = cap.read()
    image = image[:, ::-1, :]
    image_width = game.cap.get(3)

    cv_process(image, image_width, game)
    cv_output(image)



    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image, image_width, game):
    # main image processing code

    kernel = np.ones((5, 5), np.uint8)

    image_center = image_width/2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    erode = cv2.erode(hsv, kernel)

    lower = np.array([15, 50, 50])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(erode, lower, upper)

    cv2.imshow("yellow", mask)


    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image.astype(np.uint8), (x, y), (x + w, y + h), (0, 0, 255), 3)
                if (x < image_center) and (x+w < image_center):
                    game.paddle.move(-10)
                elif (x > image_center) and (x+w > image_center):
                    game.paddle.move(+10)

    pass


def cv_output(image):
    cv2.imshow("Image", image)
    # rest of output rendering
    cv2.waitKey(1)
