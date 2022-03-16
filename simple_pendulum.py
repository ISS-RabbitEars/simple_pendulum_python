import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(dv, ti, p):
	th, w = dv
	m, l, gc = p

	print(ti)
	
	return [w,alpha[0].subs({M:m, L:l, theta:th, theta.diff(t,1):w, g:gc})]


#---SymPy Derivation------------------------
L,M,g,t = sp.symbols('L M g t')
theta = dynamicsymbols('theta')

X = L * sp.sin(theta)
Y = -L * sp.cos(theta)

v_squared = X.diff(t, 1)**2 + Y.diff(t, 1)**2

T = sp.simplify(0.5 * M * v_squared)
V = M * g * Y

Lg = T - V

dLdtheta = Lg.diff(theta, 1)
dLdtheta_dot = Lg.diff(theta.diff(t, 1), 1)
ddtdLdtheta_dot = dLdtheta_dot.diff(t, 1)

diff_Lg = ddtdLdtheta_dot - dLdtheta

alpha = sp.solve(diff_Lg, theta.diff(t, 2))

#--------------------------------------------

#---functional working variables we will-----
#---use to sunbstitute into our abstract-----
#---SymPy derivation so that we can----------
#---integrate our differential equation.-----

gc = 9.8
m = 1
l = 1
theta_0 = 135 
theta_0 *= np.pi/180
omega_0 = 0

p = m, l, gc
dyn_var = theta_0, omega_0

tf = 60
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

thw = odeint(integrate, dyn_var, ta, args = (p,))

x=np.asarray([X.subs({L:l, theta:i}) for i in thw[:,0]],dtype=float)
y=np.asarray([Y.subs({L:l, theta:i}) for i in thw[:,0]],dtype=float)

ke=np.asarray([T.subs({M:m, L:l, theta.diff(t,1):i}) for i in thw[:,1]])
pe=np.asarray([V.subs({M:m, g:gc, Y:i}) for i in y])
E=ke+pe
Emax=max(E)
E/=Emax
ke/=Emax
pe/=Emax


#--aesthetics-------------------------------
xmax,ymax=l*[1.2, 1.2]
xmin,ymin=l*[-1.2, -1.2]
rad=0.05
dr=np.sqrt(x*x+y*y)
phi=np.arccos(x/dr)
dx=rad*np.cos(phi)
dy=np.sign(y)*rad*np.sin(phi)
#--plot/animation---------------------------

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	plt.plot([0,x[frame]-dx[frame]],[0,y[frame]-dy[frame]],color='xkcd:cerulean')
	circle=plt.Circle((x[frame],y[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.title("A Simple Pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('simple_pendulum.mp4', writer=writervideo)
#plt.show()


