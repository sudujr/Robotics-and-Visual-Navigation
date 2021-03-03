#!/usr/bin/env python3

import cv2
import numpy as np 
import pygame
from display import Display
import skimage


W = 1920 // 2
H = 1080 // 2
disp = Display(H,W)

class FeaturesExtract:
    def __init__(self):
        self.orb = cv2.ORB_create(100)

    def extract(self, img):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3)
        return feats

fe = FeaturesExtract()

def process_frame(image):
    frame = cv2.resize(image,(W,H))
    kp = fe.extract(frame)

    for p in kp:
        u,v = map(lambda x: int(round(x)), p[0])
        cv2.circle(frame,(u,v), color=(0, 255, 0), radius = 3)
    disp.view(frame)

if __name__ == '__main__':

    view = Display(H,W)
    cap = cv2.VideoCapture("Input/test_countryroad.mp4")
    while cap.isOpened():
        ret, frame =  cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break


