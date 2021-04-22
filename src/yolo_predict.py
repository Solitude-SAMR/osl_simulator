#!/usr/bin/env python
import rospy 
import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            # print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i] 
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)

class DetectionYolo():
    def __init__(self):
        pass

class Prediction():
    def __init__(self):
        cam_topic = "/bluerov2/camera_out/image_raw"
        self.sub_camera = rospy.Subscriber(cam_topic, Image, self.callback_camera, queue_size=1)  
        self.cv_bridge = CvBridge()
        self.img = np.zeros((1,1,3))
        
        # TODO: Make a Yolo class for these variables. 
        path_cfg = "/home/bvibhav/Documents/Solitude/wavetank/yolov3.cfg"
        path_weights = "/home/bvibhav/Documents/Solitude/wavetank/yolov3_final.weights"
        path_classes = "/home/bvibhav/Documents/Solitude/wavetank/data/classes.txt"
        self.net = cv2.dnn.readNet(path_weights, path_cfg)
        self.classes = []
        with open(path_classes, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        rospy.loginfo("Loaded Yolo Predictor.")

        # Labelled image publisher
        self.pub_img = rospy.Publisher("/bluerov2/bbox_pred", Image, queue_size=10)

        # Wait for image to arrive
        while True:
            rospy.sleep(.1)
            if self.img.shape[0]>1:
                break

    def callback_camera(self, msg_img):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg_img, "bgr8")
    
    def process(self, show_img=True):
        image = self.img.copy()
        height, width, _ = image.shape

        blob, outputs = detect_objects(image, self.net, self.output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, self.colors, class_ids, self.classes, image)

        msg_image = self.cv_bridge.cv2_to_imgmsg(image, encoding="passthrough")
        self.pub_img.publish(msg_image)

        if show_img:
            cv2.imshow('Image', image)
            key = cv2.waitKey(1)
            if key==ord('q') or key==27:
                exit()

def main():
    rospy.init_node('Predictor')
    predictor = Prediction()
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        predictor.process()
        rate.sleep()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass