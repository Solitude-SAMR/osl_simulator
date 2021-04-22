#!/usr/bin/env python
import rospy 
import numpy as np
import cv2
import tf
import signal

from datetime import datetime
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, euler_matrix

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from osl_simulator.msg import Box2DArray

np.set_printoptions(precision=4, suppress=True)

# Homogenous coordinate conversion
def get_transform(tf_listener_obj, target, source, axes='sxyz'):
    for k in range (0,10):
        try:
            (tf_trans, tf_rot) = tf_listener_obj.lookupTransform(target, source, rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(.1)
            # if (k+1)%5==0:
            rospy.logwarn('get_transform: TF from %s to %s not recieved, try %d/%d', source, target, k+1, 10)
            continue

    tf_rot = euler_from_quaternion((tf_rot))
    tf_rot =  euler_matrix(tf_rot[0], tf_rot[1], tf_rot[2], axes=axes)
    P_source_target = tf_rot
    P_source_target[0:3,3] = tf_trans
    # P_inv = tf.transformations.inverse_matrix(P_source_target)
    return P_source_target

class BoundingBox():
    def __init__(self):
        # cam_topic = "/gazebo/camera/image_raw"
        cam_topic = "/bluerov2/camera_out/image_raw"
        self.sub_bbox = rospy.Subscriber("/bbox_collated", Box2DArray, self.callback_bbox, queue_size=1)
        self.sub_camera = rospy.Subscriber(cam_topic, Image, self.callback_camera, queue_size=1)  
        self.gazebo_states = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_states)
        self.cv_bridge = CvBridge()
        self.img = np.zeros((1,1,3))
        self.img_bbox = np.zeros((1,1,3))
        self.msg_bbox = Box2DArray()

        self.model_names = ['rust_pipe', 'ball', 'gas_cannister', 'gas_tank', 'oil_drum', 'cage']
        pose_template = Pose(); pose_template.position.z = -1000
        self.model_states = [pose_template]*len(self.model_names)
        self.tf_listener = tf.TransformListener()

        self.path_images = "/home/bvibhav/ros_uuv_ws/data/images/"
        self.path_labels = "/home/bvibhav/ros_uuv_ws/data/labels/"
        self.file_classes = "/home/bvibhav/ros_uuv_ws/data/classes.txt"
        
        f_classes = open(self.file_classes, 'w')
        for c_name in self.model_names:
            f_classes.write("%s\n" % c_name)
        f_classes.close()

        # Ctrl handler
        signal.signal(signal.SIGINT, self.signal_handler)

        # Labelled image publisher
        self.pub_img = rospy.Publisher("/bluerov2/bbox_gt", Image, queue_size=10)
    
    def quit_script(self):
        exit()

    def signal_handler(self, sig, frame):
        self.quit_script()

    def callback_states(self, msg):
        for c_name in self.model_names:
            if c_name in msg.name:
                self.model_states[self.model_names.index(c_name)] = msg.pose[msg.name.index(c_name)]

    def callback_bbox(self, msg):
        self.msg_bbox = msg
        
    def callback_camera(self, msg_img):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg_img, "bgr8")

    def process(self, show_img=False):
        self.img_bbox = self.img.copy()
        height, width, _ = self.img_bbox.shape

        mat_states = np.zeros((4, len(self.model_states)))
        mat_states[3,:] = 1
        for k in range(0, len(self.model_states)):
            mat_states[0,k] = self.model_states[k].position.x
            mat_states[1,k] = self.model_states[k].position.y
            mat_states[2,k] = self.model_states[k].position.z

        P_world_bluerov = get_transform(self.tf_listener, 'bluerov2/base_link', 'world')
        mat_states = np.matmul(P_world_bluerov, mat_states)

        det_strings = []
        for c_box, c_name in zip(self.msg_bbox.boxes, self.msg_bbox.name):
            idx = self.model_names.index(c_name)
            if mat_states[0,idx]<0:
                continue

            center_x = c_box.center.x
            center_y = c_box.center.y
            size_x = c_box.size_x
            size_y = c_box.size_y

            minx = int(center_x - size_x/2)
            maxx = int(center_x + size_x/2)
            miny = int(center_y - size_y/2)
            maxy = int(center_y + size_y/2)

            minx = max(minx,0)
            maxx = min(maxx, width)
            miny = max(miny,0)
            maxy = min(maxy, height)

            prev_area = size_x*size_y            
            new_area = (maxx - minx)*(maxy - miny)

            if center_x<0 or center_y<0:              continue 
            if center_x>width or center_y>height:     continue 
            if (float(new_area)/float(prev_area))<.4: continue

            self.img_bbox = cv2.rectangle(self.img_bbox,(minx,miny),(maxx,maxy),(0,127,0),2)
            det_strings.append("%d %.4f %.4f %.4f %.4f" % (idx, center_x/width, center_y/height, size_x/width, size_y/height))
            # print height, width

        if show_img:
            cv2.imshow('Image', self.img_bbox)
            key = cv2.waitKey(1)
            if key==ord('s'):
                output_name = datetime.now().strftime('%s_%f')
                output_image = self.path_images+output_name+'.png'
                output_label = self.path_labels+output_name+'.txt'
                print("Saving image and labels:", output_name)

                cv2.imwrite(output_image, self.img.copy())
                f_labels = open(output_label, 'w')
                for c_string in det_strings:
                    f_labels.write("%s\n" % c_string)
                f_labels.close()

            elif key==ord('q') or key==27:
                self.quit_script()

def main():
    rospy.init_node('bounding_box')
    camera_bbox = BoundingBox()
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        camera_bbox.process()
        msg_image = camera_bbox.cv_bridge.cv2_to_imgmsg(camera_bbox.img_bbox, encoding="passthrough")
        # msg_image.header.seq = seq; seq += 1
        # msg_image.header.stamp = rospy.Time.now()
        # print msg_image.header
        camera_bbox.pub_img.publish(msg_image)
        rate.sleep()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass