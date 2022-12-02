import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlim3d([-2, 2])
ax.set_ylim3d([-2, 2])
ax.set_zlim3d([0, 13])
ax.set_xlabel('X(t)')
ax.set_ylabel('Y(t)')
ax.set_zlabel('Z(t)')
ax.set_title('Trajectory of electron for E vector along [120]')

x1 = np.array([0,1,1,0,0,1,1,0,0,1,1,0])
y1 = np.array([0,0,1,1,0,0,1,1,0,0,1,1])
z1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
dataSet1 = np.array([x1, y1, z1])
num1 = len(z1)

line1, = ax.plot3D(0,0,0)

def func1(i, dataSet):
    line1.set_xdata(dataSet[0][:i])
    line1.set_ydata(dataSet[1][:i])
    line1.set_3d_properties(dataSet[2][:i])
    return line1,

# animation.save(r'AnimationNew.mp4')

x2 = np.array([0,-1,-1,0,0,-1,-1,0,0,-1,-1,0])
y2 = np.array([0,0,-1,-1,0,0,-1,-1,0,0,-1,-1])
z2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
dataSet2 = np.array([x2, y2, z2])
num2 = len(z2)

line2, = ax.plot3D(0,0,0)

def func2(i, dataSet):
    line2.set_xdata(dataSet[0][:i])
    line2.set_ydata(dataSet[1][:i])
    line2.set_3d_properties(dataSet[2][:i])
    return line2,

def func(i, dataSet, line):
    line.set_xdata(dataSet[0][:i])
    line.set_ydata(dataSet[1][:i])
    line.set_3d_properties(dataSet[2][:i])
    return line,

animation2 = FuncAnimation(fig, func, frames=range(num2), fargs=(dataSet2,line2), interval=200, blit=False)
animation1 = FuncAnimation(fig, func, frames=range(num1), fargs=(dataSet1,line1), interval=200, blit=False)
# animation.save(r'AnimationNew.mp4')

plt.show()