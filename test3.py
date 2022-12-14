import cv2
import yolov5
import time
import numpy as np

# load model
model = yolov5.load('./yolov5n.pt')
print(model.names)
model.conf = 0.33

cap = cv2.VideoCapture(0)

ret, image = cap.read()

image = image[:, ::-1, :]

i = 0
begin_time = -1

while True:
    last_begin_time = begin_time
    begin_time = begin_read_time = time.time_ns()
    ret, image = cap.read()
    if not ret:
        break

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(imageRGB)
    output = image.copy()



    for pred in enumerate(results.pred):
        im = pred[0]
        im_boxes = pred[1]
        for *box, conf, cls in im_boxes:
            box_class = int(cls)
            conf = float(conf)
            # frameID, trackID, x, y, w, h, score,-1,-1,-1
            x = float(box[0])
            y = float(box[1])
            w = float(box[2]) - x
            h = float(box[3]) - y
            pt1 = np.array(np.round((float(box[0]), float(box[1]))), dtype=int)
            pt2 = np.array(np.round((float(box[2]), float(box[3]))), dtype=int)
            box_color = (255, 0, 0)
            if box_class == 39:
                cv2.rectangle(img=output,
                              pt1=pt1,
                              pt2=pt2,
                              color=box_color,
                              thickness=1)
                text = "{}:{:.2f}".format(results.names[box_class], conf)
                cv2.putText(img=output,
                            text=text,
                            org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=box_color,
                            thickness=1)

    cv2.imshow("YOLOv5", output)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
