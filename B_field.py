import matplotlib.pyplot as plt
import numpy as np

def B_line(z,L,k,zc,smooth=300):

    """Magnetic field in a magnetic mirror configuration."""

    conditions=[z<-(zc/2-smooth), (z>=-(zc/2-smooth))&(z<=(zc/2-smooth)), z>(zc/2-smooth)]
    values=[L/(1+np.exp(-k*(z+zc))), L, L/(1+np.exp(k*(z-zc)))]
    return np.select(conditions, values)



z=np.linspace(-1500,1500,1000)
y=np.linspace(-1,1,1000)



L=1
k=0.01
zc=1000
B=B_line(z,L,k,zc)


plt.figure()
plt.plot(z,B,label=f'k={k}, L={L}, zc={zc}')
L=0.5
B=B_line(z,L,k,zc)

print(B.shape)
# plt.plot(z,np.tanh(z))

plt.plot(z,B,label=f'k={k}, L={L}, zc={zc}')

plt.xlabel('z')
plt.ylabel('Magnetic Field B(z)')
plt.title('Magnetic Field in Magnetic Mirror Configuration')
plt.legend()
plt.grid()
plt.show()



B_vec_field=np.zeros((z.shape[0], y.shape[0],3))