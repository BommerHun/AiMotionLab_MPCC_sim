import numpy as np
import matplotlib.pyplot as plt


def paperclip_forward(r = 1.5, mirror = False):
    path_points = np.array([0,0])
    
    t = np.linspace(0,1, 20)

    for i in range(np.shape(t)[0]):
        point = np.array([0+r*np.cos(np.pi/4)*t[i], 0+r*np.sin(np.pi/4)*t[i]])
        path_points = np.vstack([path_points, point])

    path_points = path_points[1:-1, :]
    
    
    alpha = np.linspace(-np.pi/4, np.pi, 20)
    for i in range(np.shape(alpha)[0]):
        point = np.array([r*np.cos(alpha[i]), r*np.sin(alpha[i])+np.pow(2, 0.5)*r])
        path_points = np.vstack([path_points, point])

    path_points = path_points[:-1, :]

    for i in range(np.shape(t)[0]):
        point = np.array([-r, np.pow(2, 0.5)*r- (2*np.pow(2, 0.5)*r+r)*t[i]])
        path_points = np.vstack([path_points, point])
    
    if mirror == True:
        for i in range(np.shape(path_points[:,0])[0]):
            path_points[i,1] = - path_points[i,1]
            path_points[i,0] = - path_points[i,0]
       
    

    
    vel = np.ones([path_points.shape[0],1])

    
    return path_points, vel

def paperclip_backward(r = 1.5, mirror = False):
    t = np.linspace(0, 1, 20)


    path_points = np.array([-r, r*(1+np.sqrt(2))])



    #path_points = np.array([-r, -r*np.sin(np.pi/4)*2])

    for i in range(np.shape(t)[0]):
        point = np.array([-r, -r*(1+np.sqrt(2))+r*(t[i])])
        path_points = np.vstack([path_points, point])


    path_points = path_points[1:-1, :]

    alpha = np.linspace(0, np.pi/4, 20)

    for i in range(np.shape(alpha)[0]):
        point = np.array([-r*np.cos(alpha[i]), -r*np.sin(np.pi/4)*2+r*np.sin(alpha[i])])
        path_points = np.vstack([path_points, point])

    path_points = path_points[:-1, :]


    for i in range(np.shape(t)[0]):
        point = np.array([0-r*np.cos(np.pi/4)*(1-t[i]), 0-r*np.sin(np.pi/4)*(1-t[i])])
        path_points = np.vstack([path_points, point])

    if mirror == True:

        for i in range(np.shape(path_points[:,0])[0]):
            path_points[i,0] = - path_points[i,0]
            path_points[i,1] = - path_points[i,1]

    vel = np.ones([path_points.shape[0],1])


    return path_points, vel