import matplotlib.pyplot as plt
import numpy as np


class MagneticField:
    def __init__(self, k, zc,r,theta,z ):
        self.k = k
        self.zc = zc

        self.r = r
        self.theta = theta
        self.z = z
        self.R,self.Theta,self.Z=np.meshgrid(self.r,self.theta,self.z,indexing='ij')
        # self.x= self.r * np.cos(self.theta)
        # self.y= self.r * np.sin(self.theta)
        
        # print(self.x,self.y)
        # quit()

        
    def B_field_lines(self,Z,R,k,zc,smooth=300):

        """Magnetic field in a magnetic mirror configuration."""

        conditions=[Z<-(zc/2-smooth), (Z>=-(zc/2-smooth))&(Z<=(zc/2-smooth)), Z>(zc/2-smooth)]
        values=[R/(1+np.exp(-k*(Z+zc))), R, R/(1+np.exp(k*(Z-zc)))]
        return np.abs(np.select(conditions, values))
  

    def Compute_field(self):    

    
        B=self.B_field_lines(self.Z,self.R,self.k,self.zc)

        # plt.plot(self.Z[0,0,:],B[0,0,:])
        # plt.show()
        # quit()
   


        Bz,Bt,Br=np.gradient(B,self.r,self.theta,self.z)

        #normalise the vectors
        magnitude=np.sqrt(Br**2+Bt**2+Bz**2)
        magnitude[magnitude == 0] = 1

        Br/=magnitude
        Bz/=magnitude
        Bt/=magnitude

        Bz[:Br.shape[0]//2,:,:]*=-1  #inverser le sens du champ magnétique dans la moitié inférieure
        Br[:Br.shape[0]//2,:,:]*=-1  #inverser le sens du champ magnétique dans la moitié inférieure

        self.Br=Br
        self.Bt=Bt
        self.Bz=Bz



        #passage en cartésien
        X=self.R * np.cos(self.Theta)
        Y=self.R * np.sin(self.Theta)

        self.X=X
        self.Y=Y

        Bx = Br * np.cos(self.Theta) - Bt * np.sin(self.Theta)
        By = Br * np.sin(self.Theta) + Bt * np.cos(self.Theta)


        self.Bx=Bx
        self.By=By
        return Bx,By,Bz,X,Y,self.Z

    def Compute_gradient(self):
        #compute gradient de B
        dBr_dr, dBr_dtheta, dBr_dz = np.gradient(self.Br, self.r, self.theta, self.z)
        dBt_dr, dBt_dtheta, dBt_dz = np.gradient(self.Bt, self.r, self.theta, self.z)
        dBz_dr, dBz_dtheta, dBz_dz = np.gradient(self.Bz, self.r, self.theta, self.z)

        dBr_dtheta /= self.R  # Scale by 1/r
        dBt_dtheta /= self.R  # Scale by 1/r
        dBz_dtheta /= self.R  # Scale by 1/r

        gradient_tensor = np.zeros((3, 3) + self.Br.shape)  # Shape: (3, 3, Nr, Ntheta, Nz)
        gradient_tensor[0, 0] = dBr_dr
        gradient_tensor[0, 1] = dBt_dr
        gradient_tensor[0, 2] = dBz_dr
        gradient_tensor[1, 0] = dBr_dtheta-self.Bt / self.R
        gradient_tensor[1, 1] = dBt_dtheta+self.Br / self.R
        gradient_tensor[1, 2] = dBz_dtheta
        gradient_tensor[2, 0] = dBr_dz
        gradient_tensor[2, 1] = dBt_dz
        gradient_tensor[2, 2] = dBz_dz

        self.gradient_tensor=gradient_tensor

        gradient_magnitude = np.sqrt(np.sum(gradient_tensor**2, axis=(0, 1)))
        return gradient_magnitude

def new_magnetic_field(r,theta,z,B0,B1,B2):
    """Return B(x,t) as a numpy array"""
    return -B2*r*z, B2*np.zeros_like(theta), B0+B2*(z**2-0.5*r**2)
    # return -B2*x*z,-B2*y**z,B2*(z**2-0.5*x**2-0.5*y**2)+B0-2*B1*z
    return -B2*np.sqrt(x**2+y**2)*z*np.cos(Theta),-B2*np.sqrt(x**2+y**2)*z*np.sin(Theta),B2*(z**2-0.5*x**2-0.5*y**2)+B0-2*B1*z
    return B0*np.array([-x[0]*x[2],-x[1]*x[2],x[2]**2-0.5*x[0]**2-0.5*x[1]**2])

def main():

    # z=np.linspace(-2000,2000,50)
    z=np.linspace(-2,2,20)
    # r=np.linspace(0.1,1500,15)
    # theta=np.linspace(0,2*np.pi,4)

    # x=np.linspace(-1,1,10)
    # y=np.linspace(-1,1,10)

    r=np.linspace(0,1,20)
    theta=np.linspace(0,2*np.pi,20)

    R,Theta,Zcyl=np.meshgrid(r,theta,z,indexing='ij')
    # X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    X=R * np.cos(Theta)
    Y=R * np.sin(Theta)
    Z=Zcyl

    # Bx,By,Bz=new_magnetic_field(X,Y,Z,Theta,1,0,0.5)
    Br,Bt,Bzcyl=new_magnetic_field(R,Theta,Zcyl,1,0,0.5)

    Bx=Br*np.cos(Theta)-Bt*np.sin(Theta)
    By=Br*np.sin(Theta)+Bt*np.cos(Theta)
    Bz=Bzcyl

    e=Z.shape[0]//2
    fig=plt.figure()
    # ax=fig.add_subplot(111, projection='3d')
    plt.quiver(Z[e,:,:],Y[e,:,:],Bz[e,:,:],By[e,:,:],color='black',label='Magnetic Field Vectors')
    # plt.quiver(Z[:,e,:],X[:,e,:],Bz[:,e,:],Bx[:,e,:],color='black',label='Magnetic Field Vectors')
    # ax.quiver(X,Y,Z,Bx,By,Bz,color='black',label='Magnetic Field Vectors')
    # ax.quiver(R,Theta,Zcyl,Br,Bt,Bzcyl,color='black',label='Magnetic Field Vectors')
    # plt.quiver(Zcyl[:,e,:],R[:,e,:],Bzcyl[:,e,:],Br[:,e,:],color='black',label='Magnetic Field Vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    # ax.set_zlabel('z')
    plt.title('Magnetic Field Vectors in y-z plane')
    plt.legend()
    plt.grid()
    plt.show()
    quit()

    k=0.01
    zc=1000

    # magnetic_field=MagneticField(k, zc,r,theta,z)
    magnetic_field=MagneticField(k, zc,x,y,z)

    R=magnetic_field.R
    Bx,By,Bz,X,Y,Z=magnetic_field.Compute_field()
    Br=magnetic_field.Br
    gradient_magnitude=magnetic_field.Compute_gradient()

    x=np.unique(X)
    y=np.unique(Y)
    z=np.unique(Z)

    print(x.shape,y.shape,z.shape)
    print(Bx.shape,By.shape,Bz.shape)


    e=1


    plt.figure()
    plt.pcolormesh(Z[:,e,:],X[:,e,:],gradient_magnitude[:,e,:],cmap='viridis')
    plt.colorbar(label='|grad(B)|')
    plt.quiver(Z[:,e,:],X[:,e,:],Bz[:,e,:],Bx[:,e,:],color='white',label='B',width=0.003,scale=50,headwidth=3)
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('Magnetic Field and its gradient in Magnetic Mirror Configuration')
    plt.legend()
    # plt.savefig('B_field_mirror.png')
    plt.show()








    fig=plt.figure(figsize=(20,20))
    ax=fig.add_subplot(111, projection='3d')
    # ax.quiver(X[:,e,:],Y[:,e,:],Z[:,e,:],Bx[:,e,:],By[:,e,:],Bz[:,e,:],color='black',label='Magnetic Field Vectors',length=100,linewidth=2,arrow_length_ratio=0.5,normalize=True)
    ax.quiver(X[:,:,:],Y[:,:,:],Z[:,:,:],Bx[:,:,:],By[:,:,:],Bz[:,:,:],color='black',label='Magnetic Field Vectors',length=100,linewidth=2,arrow_length_ratio=0.5,normalize=True)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    # plt.show()
    quit()
    plt.figure()
    plt.pcolormesh(Z[:,e,:],R[:,e,:],gradient_magnitude[:,e,:],cmap='viridis')
    plt.colorbar(label='|grad(B)|')
    plt.quiver(Z[:,e,:],R[:,e,:],Bz[:,e,:],Br[:,e,:],color='white',label='B',width=0.003,scale=50,headwidth=3)
    plt.xlabel('z')
    plt.ylabel('r')
    plt.title('Magnetic Field and its gradient in Magnetic Mirror Configuration')
    plt.legend()
    # plt.savefig('B_field_mirror.png')
    plt.show()





if __name__ == "__main__":
    main()