#!/usr/bin/env python
import rospy 
import numpy as np

from vision_msgs.msg import Detection2D
from vision_msgs.msg import BoundingBox2D
from gazebo_bbox.msg import Box2DArray

class CollateBoxes():
    def __init__(self):
        self.sub_bbox = rospy.Subscriber("/bounding_box", Detection2D, self.callback_bbox, queue_size=1)
        self.pub_bbox = rospy.Publisher("/bbox_collated", Box2DArray, queue_size=1)  

        self.model_names = []
        self.detections = Box2DArray()

    def callback_bbox(self, msg):
        if not msg.header.frame_id in self.detections.name:
            self.detections.name.append(msg.header.frame_id)
            self.detections.boxes.append(msg.bbox)

        c_idx = self.detections.name.index(msg.header.frame_id)
        self.detections.boxes[c_idx] = msg.bbox

    def collate(self):
        self.detections.header.stamp = rospy.Time.now()
        self.pub_bbox.publish(self.detections)

def main():
    rospy.init_node('collate_bbox')
    collater = CollateBoxes()
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        collater.collate()
        rate.sleep()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass