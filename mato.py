import cvxpy as cp
from cvxpy.atoms.norm import norm2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


class traj:
    def __init__(self):
        self.n = 4
        self.g = 9.8
        self.T = 88
        self.Th = 8
        self.dt1 = 0.1
        self.vmax = 50
        self.umax = 50
        self.rcol = 0.5
        self.eps = 0.1
        self.lxy = 0.8
        self.lz = 0.4

        self.x1_i = np.array([-2,4,1,0,0,0])
        self.x2_i = np.array([-2,2,1,0,0,0])
        self.x3_i = np.array([2,2,1,0,0,0])
        self.x4_i = np.array([2,4,1,0,0,0])
        self.x1_f = np.array([2,0,1,0,0,0])
        self.x2_f = np.array([2,6,1,0,0,0])
        self.x3_f = np.array([-2,6,1,0,0,0])
        self.x4_f = np.array([-2,0,1,0,0,0])

        self.k = 0
        self.A = np.array([[1,0,0,self.dt1,0,0],[0,1,0,0,self.dt1,0],[0,0,1,0,0,self.dt1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.B = np.array([[self.dt1**2/2,0,0],[0,self.dt1**2/2,0],[0,0,self.dt1**2/2],[self.dt1,0,0],[0,self.dt1,0],[0,0,self.dt1]])
        self.Z = np.array([0,0,-self.g*(self.dt1**2/2),0,0,-self.g*self.dt1])
        self.C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        self.D = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        
    def opt_wo_con(self):
        x1 = cp.Variable((6,self.T+1))
        u1 = cp.Variable((3,self.T))
        x2 = cp.Variable((6,self.T+1))
        u2 = cp.Variable((3,self.T))
        x3 = cp.Variable((6,self.T+1))
        u3 = cp.Variable((3,self.T))
        x4 = cp.Variable((6,self.T+1))
        u4 = cp.Variable((3,self.T))
        alpha = cp.Variable((6,self.T+1))
        beta = cp.Variable((6,self.T+1))
        d = cp.Variable((6,self.T+1))

        cost = 0
        constr = []
        for t in range(self.T):
            cost += cp.norm2(u1[:,t])
            cost += cp.norm2(u2[:,t])
            cost += cp.norm2(u3[:,t])
            cost += cp.norm2(u4[:,t])

            constr += [x1[:,t+1] == self.A@x1[:,t] + self.B@u1[:,t]]
            constr += [x2[:,t+1] == self.A@x2[:,t] + self.B@u2[:,t]]
            constr += [x3[:,t+1] == self.A@x3[:,t] + self.B@u3[:,t]]
            constr += [x4[:,t+1] == self.A@x4[:,t] + self.B@u4[:,t]]

        # for t in range(self.T+1):
        #     constr += [0.1 <= x1[0,t]-x2[0,t]]
        #     constr += [0.1 <= x1[0,t]-x3[0,t]]
        #     constr += [0.1 <= x1[0,t]-x4[0,t]]
        #     constr += [0.1 <= x2[0,t]-x3[0,t]]
        #     constr += [0.1 <= x2[0,t]-x4[0,t]]
        #     constr += [0.1 <= x3[0,t]-x4[0,t]]

        #     constr += [0.1 <= x1[1,t]-x2[1,t]]
        #     constr += [0.1 <= x1[1,t]-x3[1,t]]
        #     constr += [0.1 <= x1[1,t]-x4[1,t]]
        #     constr += [0.1 <= x2[1,t]-x3[1,t]]
        #     constr += [0.1 <= x2[1,t]-x4[1,t]]
        #     constr += [0.1 <= x3[1,t]-x4[1,t]]

        #     constr += [0.1 <= x1[2,t]-x2[2,t]]
        #     constr += [0.1 <= x1[2,t]-x3[2,t]]
        #     constr += [0.1 <= x1[2,t]-x4[2,t]]
        #     constr += [0.1 <= x2[2,t]-x3[2,t]]
        #     constr += [0.1 <= x2[2,t]-x4[2,t]]
        #     constr += [0.1 <= x3[2,t]-x4[2,t]]

        # for t in range(self.T+1):
            constr += [beta[:,t]<= np.pi]
            constr += [0 <= beta[:,t]]
            constr += [alpha[:,t]<= np.pi]
            constr += [-np.pi <= alpha[:,t]]
            constr += [1 <= d[:,t]]
            constr += [x1[0,t]-x2[0,t]-(self.lxy*d[0,t]*(beta[0,t])) ==0]
            constr += [x1[0,t]-x3[0,t]-(self.lxy*d[1,t]*(beta[1,t])) ==0]
            constr += [x1[0,t]-x4[0,t]-(self.lxy*d[2,t]*(beta[2,t])) ==0]
            constr += [x2[0,t]-x3[0,t]-(self.lxy*d[3,t]*(beta[3,t])) ==0]
            constr += [x2[0,t]-x4[0,t]-(self.lxy*d[4,t]*(beta[4,t])) ==0]
            constr += [x3[0,t]-x4[0,t]-(self.lxy*d[5,t]*(beta[5,t])) ==0]

            constr += [x1[1,t]-x2[1,t]-(self.lxy*d[0,t]*(beta[0,t])) ==0]
            constr += [x1[1,t]-x3[1,t]-(self.lxy*d[1,t]*(beta[1,t])) ==0]
            constr += [x1[1,t]-x4[1,t]-(self.lxy*d[2,t]*(beta[2,t])) ==0]
            constr += [x2[1,t]-x3[1,t]-(self.lxy*d[3,t]*(beta[3,t])) ==0]
            constr += [x2[1,t]-x4[1,t]-(self.lxy*d[4,t]*(beta[4,t])) ==0]
            constr += [x3[1,t]-x4[1,t]-(self.lxy*d[5,t]*(beta[5,t])) ==0]

            constr += [x1[2,t]-x2[2,t]-(self.lxy*d[0,t]*(beta[0,t])) ==0]
            constr += [x1[2,t]-x3[2,t]-(self.lxy*d[1,t]*(beta[1,t])) ==0]
            constr += [x1[2,t]-x4[2,t]-(self.lxy*d[2,t]*(beta[2,t])) ==0]
            constr += [x2[2,t]-x3[2,t]-(self.lxy*d[3,t]*(beta[3,t])) ==0]
            constr += [x2[2,t]-x4[2,t]-(self.lxy*d[4,t]*(beta[4,t])) ==0]
            constr += [x3[2,t]-x4[2,t]-(self.lxy*d[5,t]*(beta[5,t])) ==0]

            # constr += [cp.norm2(self.D@x2[:,t])<= self.vmax]
            # constr += [cp.norm2(self.D@x3[:,t])<= self.vmax]
            # constr += [cp.norm2(self.D@x4[:,t])<= self.vmax]

            # constr += [cp.norm_inf(u1[:,t])<=self.umax]
            # constr += [cp.norm_inf(u2[:,t])<=self.umax]
            # constr += [cp.norm_inf(u3[:,t])<=self.umax]
            # constr += [cp.norm_inf(u4[:,t])<=self.umax]

        constr += [x1[:,0] == self.x1_i[0], x1[:,self.T] == self.x1_f]
        constr += [x2[:,0] == self.x1_i[1], x2[:,self.T] == self.x2_f]
        constr += [x3[:,0] == self.x1_i[2], x3[:,self.T] == self.x3_f]
        constr += [x4[:,0] == self.x1_i[3], x4[:,self.T] == self.x4_f]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.SCS)
        print("status:", problem.status)
        print("optimal value", problem.value)
        return x1.value, x2.value, x3.value, x4.value

    def plot(self,trajs):
            print(trajs)
            x = np.array(trajs)
            x1 = x[:,0]
            # x1 = np.hstack(xa)[:3,:]
            x2 = x[:,1]
            # x2 = np.hstack(xa)[:3,:]
            x3 = x[:,2]
            # x3 = np.hstack(xa)[:3,:]
            x4 = x[:,3]
            # x4 = np.hstack(xa)[:3,:]

            fig = plt.figure()
            ax = Axes3D(fig)

            ax.set_xlim3d([-5, 5])
            ax.set_ylim3d([-0.5, 4.5])
            ax.set_zlim3d([-1, 3])
            ax.set_xlabel('X(t)')
            ax.set_ylabel('Y(t)')
            ax.set_zlabel('Z(t)')
            ax.set_title('Trajectory of Agents')

            dataSet1 = np.array([x1[0], x1[1], x1[2]])
            dataSet2 = np.array([x2[0], x2[1], x2[2]])
            dataSet3 = np.array([x3[0], x3[1], x3[2]])
            dataSet4 = np.array([x4[0], x4[1], x4[2]])
            num = len(x1[2])

            line1, = ax.plot3D(0,0,0,"blue")
            line2, = ax.plot3D(0,0,0,"orange")
            line3, = ax.plot3D(0,0,0,"green")
            line4, = ax.plot3D(0,0,0,"red")
            point1, = ax.plot3D(0,0,1,"blue", marker="o")
            point2, = ax.plot3D(0,0,1,"orange", marker="o")
            point3, = ax.plot3D(0,0,1,"green", marker="o")
            point4, = ax.plot3D(0,0,1,"red", marker="o")

            ax.scatter3D(self.x1_f[0], self.x1_f[1], self.x1_f[2], color = "blue", marker="^")
            ax.scatter3D(self.x2_f[0], self.x2_f[1], self.x2_f[2], color = "orange", marker="^")
            ax.scatter3D(self.x3_f[0], self.x3_f[1], self.x3_f[2], color = "green", marker="^")
            ax.scatter3D(self.x4_f[0], self.x4_f[1], self.x4_f[2], color = "red", marker="^")

            def func(i):
                line1.set_xdata(dataSet1[0][:i])
                line1.set_ydata(dataSet1[1][:i])
                line1.set_3d_properties(dataSet1[2][:i])
                point1.set_xdata(dataSet1[0][i-1])
                point1.set_ydata(dataSet1[1][i-1])
                point1.set_3d_properties(dataSet1[2][i-1])
                line2.set_xdata(dataSet2[0][:i])
                line2.set_ydata(dataSet2[1][:i])
                line2.set_3d_properties(dataSet2[2][:i])
                point2.set_xdata(dataSet2[0][i-1])
                point2.set_ydata(dataSet2[1][i-1])
                point2.set_3d_properties(dataSet2[2][i-1])
                line3.set_xdata(dataSet3[0][:i])
                line3.set_ydata(dataSet3[1][:i])
                line3.set_3d_properties(dataSet3[2][:i])
                point3.set_xdata(dataSet3[0][i-1])
                point3.set_ydata(dataSet3[1][i-1])
                point3.set_3d_properties(dataSet3[2][i-1])
                line4.set_xdata(dataSet4[0][:i])
                line4.set_ydata(dataSet4[1][:i])
                line4.set_3d_properties(dataSet4[2][:i])
                point4.set_xdata(dataSet4[0][i-1])
                point4.set_ydata(dataSet4[1][i-1])
                point4.set_3d_properties(dataSet4[2][i-1])
                return line1, line2, line3, line4, point1, point2, point3, point4

            animation = FuncAnimation(fig, func, frames=range(num), interval=200, blit=False,save_count=1500)
            animation.save(r'AnimationNew.mp4')
            plt.show()

if __name__ == '__main__':
    tob = traj()
    # x_actual = [tob.x1_i,tob.x2_i,tob.x3_i,tob.x4_i]
    xfin = tob.opt_wo_con()
    # print(xfin)
    tob.plot(xfin)