import cvxpy as cp
from cvxpy.atoms.norm import norm2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class traj:
    def __init__(self):
        self.n = 4
        self.g = 9.8
        self.T = 20
        self.Th = 4
        self.dt1 = 0.2
        self.dt2 = 0.5
        self.k = 0
        self.A = np.array([[1,0,0,self.dt1,0,0],[0,1,0,0,self.dt1,0],[0,0,1,0,0,self.dt1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.B = np.array([[self.dt1**2/2,0,0],[0,self.dt1**2/2,0],[0,0,self.dt1**2/2],[self.dt1,0,0],[0,self.dt1,0],[0,0,self.dt1]])
        self.Z = np.array([0,0,-self.g*(self.dt1**2/2),0,0,-self.g*self.dt1])
        self.C = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0],[0,0,0]])
        self.D = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

        #self.x = np.zeros((self.n,self.T,6))
        #self.u = np.zeros((self.n,self.T,3))
        self.vmax = 3
        self.umax = 2
        self.rcol = 0.5
        self.eps = 0.01

        self.x1_i = np.array([-4,0,0,0,0,0])
        self.x2_i = np.array([-2,0,0,0,0,0])
        self.x3_i = np.array([2,0,0,0,0,0])
        self.x4_i = np.array([4,0,0,0,0,0])
        self.x1_f = np.array([2,2,0,0,0,0])
        self.x2_f = np.array([2,4,0,0,0,0])
        self.x3_f = np.array([-2,4,0,0,0,0])
        self.x4_f = np.array([-2,2,0,0,0,0])
        
    def opt_wo_con(self,x_actual):
        x1 = cp.Variable((6,self.T+1))
        u1 = cp.Variable((3,self.T))
        x2 = cp.Variable((6,self.T+1))
        u2 = cp.Variable((3,self.T))
        x3 = cp.Variable((6,self.T+1))
        u3 = cp.Variable((3,self.T))
        x4 = cp.Variable((6,self.T+1))
        u4 = cp.Variable((3,self.T))

        cost = 0
        constr = []
        for t in range(self.T):
            cost += cp.norm2(u1[:,t])*self.dt1
            cost += cp.norm2(u2[:,t])*self.dt1
            cost += cp.norm2(u3[:,t])*self.dt1
            cost += cp.norm2(u4[:,t])*self.dt1

            constr += [x1[:,t+1] == self.A@x1[:,t] + self.B@u1[:,t]]
            constr += [x2[:,t+1] == self.A@x2[:,t] + self.B@u2[:,t]]
            constr += [x3[:,t+1] == self.A@x3[:,t] + self.B@u3[:,t]]
            constr += [x4[:,t+1] == self.A@x4[:,t] + self.B@u4[:,t]]

            constr += [cp.norm2(self.D@x1[:,t])<= self.vmax]
            constr += [cp.norm2(self.D@x2[:,t])<= self.vmax]
            constr += [cp.norm2(self.D@x3[:,t])<= self.vmax]
            constr += [cp.norm2(self.D@x4[:,t])<= self.vmax]

            constr += [cp.norm_inf(u1[:,t])<=self.umax]
            constr += [cp.norm_inf(u2[:,t])<=self.umax]
            constr += [cp.norm_inf(u3[:,t])<=self.umax]
            constr += [cp.norm_inf(u4[:,t])<=self.umax]

        constr += [x1[:,0] == x_actual[0], x1[:,self.T] == self.x1_f]
        constr += [x2[:,0] == x_actual[1], x2[:,self.T] == self.x2_f]
        constr += [x3[:,0] == x_actual[2], x3[:,self.T] == self.x3_f]
        constr += [x4[:,0] == x_actual[3], x4[:,self.T] == self.x4_f]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.SCS)
        print("status:", problem.status)
        print("optimal value", problem.value)
        return x1.value

    def opt_con(self,ko,x_actual,x_nom):
        x1 = cp.Variable((6,self.T+1-ko))
        u1 = cp.Variable((3,self.T-ko))
        x2 = cp.Variable((6,self.T+1-ko))
        u2 = cp.Variable((3,self.T-ko))
        x3 = cp.Variable((6,self.T+1-ko))
        u3 = cp.Variable((3,self.T-ko))
        x4 = cp.Variable((6,self.T+1-ko))
        u4 = cp.Variable((3,self.T-ko))

        cost = 0
        constr = []
        for t in range(self.T-ko):
            cost += cp.norm2(u1[:,t])*self.dt1
            cost += cp.norm2(u2[:,t])*self.dt1
            cost += cp.norm2(u3[:,t])*self.dt1
            cost += cp.norm2(u4[:,t])*self.dt1

            constr += [x1[:,t+1] == self.A@x1[:,t] + self.B@u1[:,t]]
            constr += [x2[:,t+1] == self.A@x2[:,t] + self.B@u2[:,t]]
            constr += [x3[:,t+1] == self.A@x3[:,t] + self.B@u3[:,t]]
            constr += [x4[:,t+1] == self.A@x4[:,t] + self.B@u4[:,t]]

            constr += [cp.norm2(self.D@x1[:,t])<= self.vmax]
            constr += [cp.norm2(self.D@x2[:,t])<= self.vmax]
            constr += [cp.norm2(self.D@x3[:,t])<= self.vmax]
            constr += [cp.norm2(self.D@x4[:,t])<= self.vmax]

            constr += [cp.norm_inf(u1[:,t])<=self.umax]
            constr += [cp.norm_inf(u2[:,t])<=self.umax]
            constr += [cp.norm_inf(u3[:,t])<=self.umax]
            constr += [cp.norm_inf(u4[:,t])<=self.umax]
        
        if ko+self.Th<self.T:
            end_t = ko+self.Th
        else:
            end_t = self.T
        for t in range(end_t-ko):
            constr += [np.transpose(x_nom[1][:,t]-x_nom[0][:,t]) @ np.transpose(self.C) @ self.C @ (x2[:,t]-x_nom[0][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[1][:,t]-x_nom[0][:,t]), ord=2) ]
            constr += [np.transpose(x_nom[2][:,t]-x_nom[0][:,t]) @ np.transpose(self.C) @ self.C @ (x3[:,t]-x_nom[0][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[2][:,t]-x_nom[0][:,t]), ord=2) ]
            constr += [np.transpose(x_nom[2][:,t]-x_nom[1][:,t]) @ np.transpose(self.C) @ self.C @ (x3[:,t]-x_nom[1][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[2][:,t]-x_nom[1][:,t]), ord=2) ]
            constr += [np.transpose(x_nom[3][:,t]-x_nom[0][:,t]) @ np.transpose(self.C) @ self.C @ (x4[:,t]-x_nom[0][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[3][:,t]-x_nom[0][:,t]), ord=2) ]
            constr += [np.transpose(x_nom[3][:,t]-x_nom[1][:,t]) @ np.transpose(self.C) @ self.C @ (x4[:,t]-x_nom[1][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[3][:,t]-x_nom[1][:,t]), ord=2) ]
            constr += [np.transpose(x_nom[3][:,t]-x_nom[2][:,t]) @ np.transpose(self.C) @ self.C @ (x4[:,t]-x_nom[2][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[3][:,t]-x_nom[2][:,t]), ord=2) ]
            #constr += [np.transpose(x_nom[1][:,t]-x_nom[0][:,t]) @ np.transpose(self.C) @ self.C @ (x2[:,t]-x_nom[0][:,t]) >= self.rcol*np.linalg.norm(self.C@(x_nom[1][:,t]-x_nom[0][:,t]), ord=2) ]


        constr += [x1[:,0] == x_actual[0], x1[:,self.T-ko] == self.x1_f]
        constr += [x2[:,0] == x_actual[1], x2[:,self.T-ko] == self.x2_f]
        constr += [x3[:,0] == x_actual[2], x3[:,self.T-ko] == self.x3_f]
        constr += [x4[:,0] == x_actual[3], x4[:,self.T-ko] == self.x4_f]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.SCS)
        return x1.value, x2.value, x3.value, x4.value

    def scp(self,ko,x_actual):
        x_old = []
        x_new = []
        k = [1,2,3,4]
        while len(k)!=0:
            x_new = self.opt_con(ko,x_actual,self.x_nom)
            self.x_nom = x_new
            b = [[None],[None],[None]]
            for j in range(4):
                for i in range(4):
                    if i<j:
                        b[j].append(np.linalg.norm(self.C@(x_new[j][:,:]-x_new[i][:,:]), ord=2)>self.rcol)######################
            for j in range(4):
                if x_new[j]-x_old[j]<self.eps and all(b[j]):
                    k.pop(j)
            x_old = x_new
        return x_old

    def mpc(self):
        ko = 0
        x_fin = []
        x_actual = [self.x1_i,self.x2_i,self.x3_i,self.x4_i]
        self.x_nom = []
        self.x_nom = self.opt_wo_con(x_actual)
        while ko<=self.T-self.Th:
            x_t = self.scp(ko,x_actual)
            curr_t = ko+self.Th
            x_actual = []
            x_semifin = []
            for i in range(4):
                x_semifin.append(x_t[i][:,ko:curr_t])
                x_actual.append(x_t[i][:,curr_t])
            x_fin.append(x_semifin)
            ko += self.Th
        while ko<self.T:
            x_t = self.scp(ko,x_actual)
            x_fin.append(x_t)
    
    def plot(self,trajs):
        x = np.array(trajs)
        xa = x[:,0]
        x1 = np.hstack(xa)[:3,:]
        xa = x[:,1]
        x2 = np.hstack(xa)[:3,:]
        xa = x[:,2]
        x3 = np.hstack(xa)[:3,:]
        xa = x[:,3]
        x4 = np.hstack(xa)[:3,:]

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_xlim3d([-2, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([0, 13])
        ax.set_xlabel('X(t)')
        ax.set_ylabel('Y(t)')
        ax.set_zlabel('Z(t)')
        ax.set_title('Trajectory of Agents')

        dataSet1 = np.array([x1[0], x1[1], x1[2]])
        dataSet2 = np.array([x2[0], x2[1], x2[2]])
        dataSet3 = np.array([x3[0], x3[1], x3[2]])
        dataSet4 = np.array([x4[0], x4[1], x4[2]])
        num = len(x1[2])

        line1, = ax.plot3D(0,0,0)
        line2, = ax.plot3D(0,0,0)
        line3, = ax.plot3D(0,0,0)
        line4, = ax.plot3D(0,0,0)

        def func(i):
            line1.set_xdata(dataSet1[0][:i])
            line1.set_ydata(dataSet1[1][:i])
            line1.set_3d_properties(dataSet1[2][:i])
            line2.set_xdata(dataSet2[0][:i])
            line2.set_ydata(dataSet2[1][:i])
            line2.set_3d_properties(dataSet2[2][:i])
            line3.set_xdata(dataSet3[0][:i])
            line3.set_ydata(dataSet3[1][:i])
            line3.set_3d_properties(dataSet3[2][:i])
            line4.set_xdata(dataSet4[0][:i])
            line4.set_ydata(dataSet4[1][:i])
            line4.set_3d_properties(dataSet4[2][:i])
            return line1, line2, line3, line4

        animation = FuncAnimation(fig, func, frames=range(num), interval=200, blit=False)
        # animation.save(r'AnimationNew.mp4')
        plt.show()

if __name__ == '__main__':
    tob = traj()
    x_actual = [tob.x1_i,tob.x2_i,tob.x3_i,tob.x4_i]
    xa = tob.opt_wo_con(x_actual)
    # print(xa[:2,:])
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlim3d([-6, -2])
    ax.set_ylim3d([0, 2])
    ax.set_zlim3d([-1, 1])
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('Z(t)')
    ax.set_title('Trajectory of electron for E vector along [120]')

    dataSet1 = np.array([xa[0,:], xa[1,:], xa[2,:]])
    num1 = len(xa[2,:])

    line1, = ax.plot3D(0,0,0)

    def func(i):
        line1.set_xdata(dataSet1[0][:i])
        line1.set_ydata(dataSet1[1][:i])
        line1.set_3d_properties(dataSet1[2][:i])
        return line1

    # def func2(i, dataSet):
    #     line2.set_xdata(dataSet[0][:i])
    #     line2.set_ydata(dataSet[1][:i])
    #     line2.set_3d_properties(dataSet[2][:i])
    #     return line2,

    # def func(i, dataSet, line):
    #     line.set_xdata(dataSet[0][:i])
    #     line.set_ydata(dataSet[1][:i])
    #     line.set_3d_properties(dataSet[2][:i])
    #     return line,

    # animation2 = FuncAnimation(fig, func, frames=range(num2), fargs=(dataSet2,), interval=200, blit=False)
    animation1 = FuncAnimation(fig, func, frames=range(num1), interval=200, blit=False)
    # animation.save(r'AnimationNew.mp4')

    plt.show()