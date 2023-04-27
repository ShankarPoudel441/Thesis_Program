import numpy as np
import pandas as pd
import math
from math import sin, cos, radians, pi

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


#Constants
rolller_length=6

#Hyper-parameters
grid_size_x=0.5
grid_size_y=0.5

#Parameters
min_x=-2
min_y=-1
max_x=2
max_y=2

#create_matrix_grid
list_x=np.arange(min_x+(grid_size_x/2),max_x,grid_size_x)
list_y=np.arange(min_y+(grid_size_y/2),max_y,grid_size_y)
list_x,list_y

xv,yv=np.meshgrid(list_x,list_y)
f=np.zeros(xv.shape)

def point_pos(x0, y0, d, theta):
    theta_rad = radians(theta)
    return x0 + d*cos(theta_rad), y0 + d*sin(theta_rad)

def given_two_point_find_rectangle(x0,y0,x1,y1,roller_length=5):
    d=roller_length/2
    theta=math.degrees(math.atan2((y1-y0),(x1-x0))) 
    print("theta made with positive x axis=",theta)
    (xa,ya)=point_pos(x0,y0,d,theta-90)
    (xb,yb)=point_pos(x0,y0,d,theta+90)
    (xc,yc)=point_pos(x1,y1,d,theta+90)
    (xd,yd)=point_pos(x1,y1,d,theta-90)
    return (xa,ya),(xb,yb),(xc,yc),(xd,yd)

given_two_point_find_rectangle(0,0,1,1)