#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import os
import copy
import cv2
import tf
import pandas as pd
import numpy as np
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2

LIDAR_DATA_PATH = "D:/Kitty_2011_09_26_0005/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data"
IMAGE_PATH = "D:/Kitty_2011_09_26_0005/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data"
TRACKELET_FILE = "D:/data_tracking_label_2/training/label_02/0000.txt"
COLUMN_NAMES = ["frame", "track_id", "type", "truncated", "occluded", "alpha", "bbox_left", "bbox_top", "bbox_right",
                "bbox_bottom", "height", "width", "length", "x", "y", "z", "yaw"]
TYPE_COLOR = {'Car': (255, 255, 0), 'Pedestrian': (0, 255, 255), 'Cyclist': (255, 0, 255)}
DATA_LENGTH = len(os.listdir(LIDAR_DATA_PATH))

LINES = [[0, 1], [1, 2], [2, 3], [3, 0]]
LINES += [[4, 5], [5, 6], [6, 7], [7, 4]]
LINES += [[4, 0], [5, 1], [6, 2], [7, 3]]
LINES += [[4, 1], [5, 0]]


# ===================================
# ==========Image and bbox===========
# ===================================
def publish_img(publisher, img, tracking_data):
    copy_img = copy.deepcopy(img)

    # 畫bbox
    bboxes = np.array(
        tracking_data[tracking_data.frame == file_id][["bbox_left", "bbox_top", "bbox_right", "bbox_bottom", "type"]])

    for bbox in bboxes:
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(copy_img, top_left, bottom_right, TYPE_COLOR[bbox[4]], 2)

    publisher.publish(bridge.cv2_to_imgmsg(copy_img, "bgr8"))


# ===================================
# =============3D lidar==============
# ===================================
def publish_lidar(publisher, point_cloud, header):
    publisher.publish(pcl2.create_cloud_xyz32(header, point_cloud[:, :3]))


# ===================================
# =============3D bbox===============
# ===================================
def publish_3dbbox(publisher, file_id, tracking_data):
    all_3dbbox = []
    obj_type = []

    # 讀取3dbbox資訊
    _3dbboxes = np.array(
        tracking_data[tracking_data.frame == file_id][["height", "width", "length", "x", "y", "z", "yaw", "type"]])

    # 設定8個頂點座標
    for _3dbbox in _3dbboxes:
        h, w, l, x, y, z, yaw, type = _3dbbox
        x_corners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]])
        y_corners = np.array([[0, 0, 0, 0, -h, -h, -h, -h]])
        z_corners = np.array([[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]])

        # 將其進行旋轉
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])

        corners = R @ np.vstack([x_corners, y_corners, z_corners])

        # 將旋轉完後的頂點加回中心點
        corners += np.array([[x],
                             [y],
                             [z]])

        # 將座標轉換到lidar座標系下
        P = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                      [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                      [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                      [0, 0, 0, 1]])
        P_inv = np.linalg.inv(P)
        corners = np.vstack([corners, np.ones(shape=(1, 8))])  # shape (4, 8)
        corners = P_inv @ corners  # shape (4, 8)
        corners = corners[:3, :] / corners[-1, :]  # shape (3, 8)

        all_3dbbox.append(corners)
        obj_type.append(type)

    # 發布
    marker_array = MarkerArray()
    for i, _3dbbox in enumerate(all_3dbbox):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.action = Marker.ADD
        marker.type = Marker.LINE_LIST
        marker.lifetime = rospy.Duration(0.1)
        marker.color.r = TYPE_COLOR[obj_type[i]][0] / 255.
        marker.color.g = TYPE_COLOR[obj_type[i]][1] / 255.
        marker.color.b = TYPE_COLOR[obj_type[i]][2] / 255.
        marker.color.a = 1.0
        marker.scale.x = 0.1
        marker.points = []

        detect_points_set = []
        for i in range(8):
            detect_points_set.append(Point(_3dbbox[0, i], _3dbbox[1, i], _3dbbox[2, i]))

        for line in LINES:
            marker.points.append(detect_points_set[line[0]])
            marker.points.append(detect_points_set[line[1]])

        marker_array.markers.append(marker)
    publisher.publish(marker_array)


# ===================================
# =========Lidar with img============
# ===================================
def publish_lidar_with_img(publisher, img, point_cloud):
    copy_img = copy.deepcopy(img)

    point_cloud = point_cloud[:, :3].T  # shape (3, N)

    # 轉換至齊次坐標系下
    point_cloud = np.vstack([point_cloud, np.ones(shape=(1, point_cloud.shape[1]))])  # shape (4, N)

    # 將lidar坐標系轉換到2號相機坐標系 (P02 @ R0 @ vel_to_cam)
    P = np.array([[6.09695409e+02, -7.21421597e+02, -1.25125855e+00, -1.23041806e+02],
                  [1.80384202e+02, 7.64479802e+00, -7.19651474e+02, -1.01016688e+02],
                  [9.99945389e-01, 1.24365378e-04, 1.04513030e-02, -2.69386912e-01],
                  [0, 0, 0, 1]])

    point_cloud = P @ point_cloud  # shape (4, N)

    # 每間隔5個取一點，加快執行速度
    point_cloud = point_cloud[:, ::10]

    # 正規化
    point_cloud[:2, :] /= point_cloud[2, :]  # shape (4, N)

    # 去除大於相機座標的點，還有距離小於0的座標
    img_height, img_width, _ = img.shape
    mask = np.where((point_cloud[0, :] > 0) & (point_cloud[0, :] < img_width) & (point_cloud[1, :] > 0) & (
            point_cloud[1, :] < img_height) & (point_cloud[2, :] > 0))
    col = point_cloud[0, mask].reshape((-1,))
    row = point_cloud[1, mask].reshape((-1,))
    depth = point_cloud[2, mask].reshape((-1,))

    # 畫出點雲，距離越近顏色越紅，越遠則越藍
    for x, y, z in zip(col, row, depth):
        b = z * 255 / 20
        r = 255 - b
        cv2.circle(copy_img, (int(x), int(y)), 2, (b, 0, r), -1)

    publisher.publish(bridge.cv2_to_imgmsg(copy_img, "bgr8"))


if __name__ == "__main__":
    rospy.init_node("Visualization_Node", anonymous=True)

    # 設定發布器
    lidar_publisher = rospy.Publisher("/Lidar", PointCloud2, queue_size=10)
    image_publisher = rospy.Publisher("/Image", Image, queue_size=10)
    car_model_publisher = rospy.Publisher("/Car_model", Marker, queue_size=10)
    _3dbbox_publisher = rospy.Publisher("/_3dbbox", MarkerArray, queue_size=10)
    lidar_with_img_publisher = rospy.Publisher("/Lidar_with_Img", Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(10)
    file_id = 0

    # 追蹤數據
    tracking_data = pd.read_csv(TRACKELET_FILE, sep=' ')
    tracking_data.columns = COLUMN_NAMES
    tracking_data.loc[tracking_data.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    tracking_data = tracking_data[tracking_data.type.isin(['Car', 'Pedestrian', 'Cyclist'])]

    while not rospy.is_shutdown():
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'

        # 讀取資料
        rgb_img = cv2.imread((IMAGE_PATH + "/" + "%010d.png" % file_id))
        point_cloud = np.fromfile(LIDAR_DATA_PATH + "/" + "%010d.bin" % file_id, dtype=np.float32).reshape(-1, 4)

        publish_img(image_publisher, rgb_img, tracking_data)
        publish_lidar(lidar_publisher, point_cloud, header)
        publish_3dbbox(_3dbbox_publisher, file_id, tracking_data)
        publish_lidar_with_img(lidar_with_img_publisher, rgb_img, point_cloud)

        rate.sleep()
        file_id += 1
        file_id %= DATA_LENGTH
