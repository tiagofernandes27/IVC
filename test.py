import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    kernel = np.ones((5, 5), np.uint8)

    # dilate = cv2.dilate(image, kernel)
    # cv2.imshow("Dilate", dilate)

    # erode = cv2.erode(image, kernel)
    # cv2.imshow("Erode", erode)

    # image.width
    image_width = cap.get(3)
    # print(image_width)

    # edges = cv2.Canny(image, 100, 200)
    # cv2.imshow("Edges", edges)

    # inverte imagem / espelhado
    # image = image[:, ::-1, :]

    # transforma imagem em HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv", hsv)

    # dilate = cv2.dilate(hsv, kernel)
    # cv2.imshow("Dilate", dilate)

    erode = cv2.erode(hsv, kernel)
    # cv2.imshow("Erode", erode)

    # region Colors

    # Red
    # lower = np.array([0, 50, 50])
    # upper = np.array([10, 255, 255])
    # mask1 = cv2.inRange(hsv, lower, upper)
    # lower = np.array([170, 50, 50])
    # upper = np.array([180, 255, 255])
    # mask2 = cv2.inRange(hsv, lower, upper)
    # mask = mask1 + mask2

    # Yellow
    lower = np.array([15, 50, 50])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(erode, lower, upper)

    # Blue
    # lower = np.array([90, 50, 50])
    # upper = np.array([130, 255, 255])
    # mask = cv2.inRange(hsv, lower, upper)

    # Green
    # lower = np.array([46, 25, 25])
    # upper = np.array([70, 255, 255])
    # mask = cv2.inRange(hsv, lower, upper)

    # cv2.imshow("green", mask)

    # endregion

    result = cv2.bitwise_and(image, image, mask=mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
                if (x < image_width/2) and (x+w < image_width/2):
                    print("right")
                elif (x > image_width/2) and (x+w > image_width/2):
                    print("left")

    image = image[:, ::-1, :]
    cv2.imshow("image", image)
    mask = mask[:, ::-1]
    cv2.imshow("yellow", mask)
    # cv2.imshow("only yellow", result)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
