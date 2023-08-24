import numpy as np
from types import FunctionType

class ODE_Solver:
    """Class of various numerical explicit methods to solve
    explicit ordinary differential equations (ODE). ODE needs to
    be reduced to first oder (dx/dt = Ax+b(t,x))
    Refer to https://en.wikipedia.org/wiki/Ordinary_differential_equation#Reduction_to_a_first-order_system
    and https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    def __init__(self, A, b):
        assert type(A)==np.ndarray
        assert A.ndim > 1
        self.A = A
        self.b = b

    def run_Euler(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k = self.A@x[n]+self.b(t[n],x[n])
            x_next = x[n]+h*k
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x),t
    
    def run_Explicit_mid(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h/2*k1)+self.b(t[n]+h/2,x[n]+h/2*k1)
            x_next = x[n]+h*k2
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t

    def run_Heun(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*k1)+self.b(t[n]+h,x[n]+h*k1)
            x_next = x[n]+h/2*(k1+k2)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t

    def run_Ralston(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*2/3*k1)\
                +self.b(t[n]+2/3*h,x[n]+h*2/3*k1)
            x_next = x[n]+h*(1/4*k1+3/4*k2)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t

    def run_general_2nd(self,alpha,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        assert alpha != 0
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*alpha*k1)\
                +self.b(t[n]+alpha*h,x[n]+h*alpha*k1)
            x_next = x[n]+h*((1-1/(2*alpha))*k1+1/(2*alpha)*k2)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t
    
    def run_K3(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h/2*k1)+self.b(t[n]+h/2,x[n]+h/2*k1)
            k3 = self.A@(x[n]-h*k1+2*h*k2)\
                +self.b(t[n]+h,x[n]-h*k1+2*h*k2)
            x_next = x[n]+h*(1/6*k1+2/3*k2+1/6*k3)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t
    
    def run_general_3rd(self,alpha,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        assert alpha != 0 and alpha != 2/3 and alpha != 1
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*alpha*k1)\
                +self.b(t[n]+h*alpha,x[n]+h*alpha*k1)
            k3 = self.A@(x[n]+h*((1+(1-alpha)/(alpha*(3*alpha-2)))*k1\
                                 -(1-alpha)/(alpha*(3*alpha-2))*k2))\
                +self.b(t[n]+h,x[n]+h*((1+(1-alpha)/(alpha*(3*alpha-2)))*k1\
                                 -(1-alpha)/(alpha*(3*alpha-2))*k2))
            x_next = x[n]+h*((1/2-1/(6*alpha))*k1+1/(6*alpha*(1-alpha))*k2+\
                             (2-3*alpha)/(6*(1-alpha))*k3)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t

    def run_Van_Houwen(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*8/15*k1)+self.b(t[n]+h*8/15,x[n]+h*8/15*k1)
            k3 = self.A@(x[n]+h*1/4*k1+h*5/12*k2)\
                +self.b(t[n]+h*2/3,x[n]+h*1/4*k1+h*5/12*k2)
            x_next = x[n]+h*(1/4*k1+3/4*k3)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t
    
    def run_Ralston_3rd(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h/2*k1)+self.b(t[n]+h/2,x[n]+h/2*k1)
            k3 = self.A@(x[n]+h*3/4*k2)\
                +self.b(t[n]+h*3/4,x[n]+h*3/4*k2)
            x_next = x[n]+h*(2/9*k1+1/3*k2+4/9*k3)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t

    def run_SSPRK3(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*k1)+self.b(t[n]+h,x[n]+h*k1)
            k3 = self.A@(x[n]+h*1/4*k1+h*1/4*k2)\
                +self.b(t[n]+h/2,x[n]+h*1/4*k1+h*1/4*k2)
            x_next = x[n]+h*(1/6*k1+1/6*k2+2/3*k3)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x), t
    
    def run_RK4(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h/2*k1)+self.b(t[n]+h/2,x[n]+h/2*k1)
            k3 = self.A@(x[n]+h/2*k2)+self.b(t[n]+h/2,x[n]+h/2*k2)
            k4 = self.A@(x[n]+h*k3)+self.b(t[n]+h,x[n]+h*k3)
            x_next = x[n]+h/6*(k1+2*k2+2*k3+k4)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x),t

    def run_3_eight_4th(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h/3*k1)+self.b(t[n]+h/3,x[n]+h/3*k1)
            k3 = self.A@(x[n]-h/3*k1+h*k2)\
                +self.b(t[n]+h*2/3,x[n]-h/3*k1+h*k2)
            k4 = self.A@(x[n]+h*k1-h*k2+h*k3)\
                +self.b(t[n]+h,x[n]+h*k1-h*k2+h*k3)
            x_next = x[n]+h/8*(k1+3*k2+3*k3+k4)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x),t

    def run_Ralston_4th(self,x0,t0,h,N_steps):
        assert type(x0)==np.ndarray
        x,t = [list(x0)],[t0]
        for n in range(N_steps):
            k1 = self.A@x[n]+self.b(t[n],x[n])
            k2 = self.A@(x[n]+h*0.4*k1)+self.b(t[n]+h*0.4,x[n]+h*0.4*k1)
            k3 = self.A@(x[n]+h*(0.29697761*k1+0.15875964*k2))\
                +self.b(t[n]+h*0.45573725,x[n]+h*(0.29697761*k1+0.15875964*k2))
            k4 = self.A@(x[n]+h*0.21810040*k1-h*3.05096516*k2+h*3.83286476*k3)\
                +self.b(t[n]+h,x[n]+h*0.21810040*k1-h*3.05096516*k2+h*3.83286476*k3)
            x_next = x[n]+h*(0.17476028*k1-0.55148066*k2\
                             +1.20553560*k3+0.17118478*k4)
            x.append(list(x_next))
            t.append(t[n]+h)
        return np.array(x),t