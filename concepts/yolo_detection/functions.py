import cv2
import numpy as np
import matplotlib.pyplot as plt

kLeftBottomGap = 120
kLeftTopGap = 600
kTopGap = 200
resolution = [1280, 720]
outputSize = [300, 600]
leftTop = [kLeftTopGap,kTopGap]
rightTop = [resolution[0]-kLeftTopGap,kTopGap]
leftBottom = [kLeftBottomGap,resolution[1]]
rightBottom = [resolution[0]-kLeftBottomGap,resolution[1]]

def changePerspective(img):
    # LT, RT, LB, RB
    pts = [leftTop,rightTop,leftBottom,rightBottom]

    pts1 = np.float32(pts)
    pts2 = np.float32([[0,0],[outputSize[0],0],[0,outputSize[1]],[outputSize[0],outputSize[1]]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,outputSize)

    colour = (0, 255, 0)
    thickness = 3
    img = cv2.line(img, leftBottom, leftTop, colour, thickness)
    img = cv2.line(img, rightBottom, rightTop, colour, thickness)

    return img, dst

def convertBinary(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img,(5,5))
    thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)[-1]
    return thresh

def adaptiveThresholding(image, blur):
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    img = cv2.medianBlur(gray,blur)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    return th2, th3

def colourThresholdingHSV(image):
    image = image[image.shape[0] - int(image.shape[0]/10):, :]
    image = cv2.pyrUp(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    huL = 30
    huH = 179
    saL = 35
    saH = 255
    vaL = 0
    vaH = 255
    HSVLOW = np.array([huL, saL, vaL])
    HSVHIGH = np.array([huH, saH, vaH])

    # apply the range on a mask
    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
    cv2.imshow('mask', mask)
    maskedFrame = cv2.bitwise_and(image, image, mask = mask)
    
    return maskedFrame, image

def getContours(image):
    objColor = (0,0,255)
    marked = image.copy()
    ratio = 30
    start_dist = 1.5
    
    # Count the contours on masked frame
    kernel = np.ones((5,5),np.uint8)
    masked = cv2.dilate(image, kernel, iterations = 1)
    masked  = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    Contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rangeCount = 0

    for i in range (0, len(Contours)):
        cnt = Contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        
        distance = ratio - (y+h)*ratio/marked.shape[0] + start_dist
        distance = "{:.2f}".format(distance)
        rangeCount = rangeCount + 1
        cv2.drawContours(marked, [cnt], -1, objColor, 3)
        marked = cv2.rectangle(marked, (x, y), (x+w, y+h), (255,255,255), 2)
        marked = cv2.putText(marked, str(distance), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        marked = cv2.putText(marked, str(distance), (x-50, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return marked, rangeCount

def getCoords(event,x,y,flags,img):

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)
        cv2.circle(img,(x,y),3,(255,0,0),-1)

def centerLines(image):
    # image = cv2.rectangle(image, (0, int(image.shape[0]/2)), (image.shape[1], int(image.shape[0]/2)), (0,0,0), 2)
    # image = cv2.rectangle(image, (int(image.shape[1]/2), 0), (int(image.shape[1]/2), image.shape[0]), (0,0,0), 2)
    return image

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):

    label = str(classes[class_id])
    color = COLORS[class_id]
    h = img.shape[1]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # cv2.line(img, (x,y_plus_h), (x,h), (255,255,255), 1)

    # print('Distance: ' + str(h-y_plus_h) + ' pixels')

def yolo(frame):
    image = frame.copy()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None

    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    maskedImage = np.zeros((image.shape[0],image.shape[1],3), np.uint8)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)
        maskedImage = cv2.rectangle(maskedImage, (round(x),round(y)), (round(x+w),round(y+h)), (255,255,255), -1)

    return image, maskedImage

if __name__ == '__main__':
    pass