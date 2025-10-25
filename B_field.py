import matplotlib.pyplot as plt
import numpy as np


class MagneticField:
    def __init__(self, k, zc,r,theta,z ):
        self.k = k
        self.zc = zc
        self.r = r
        self.theta = theta
        self.z = z
        self.R,self.Theta,self.Z=np.meshgrid(r,theta,z,indexing='ij')

        
    def B_field_lines(self,Z,R,k,zc,smooth=300):

        """Magnetic field in a magnetic mirror configuration."""

        conditions=[Z<-(zc/2-smooth), (Z>=-(zc/2-smooth))&(Z<=(zc/2-smooth)), Z>(zc/2-smooth)]
        values=[R/(1+np.exp(-k*(Z+zc))), R, R/(1+np.exp(k*(Z-zc)))]
        return np.abs(np.select(conditions, values))
  

    def Compute_field(self):    
        B=self.B_field_lines(self.Z,self.R,self.k,self.zc)


        Bz,Bt,Br=np.gradient(B,self.r,self.theta,self.z)

        #normalise the vectors
        magnitude=np.sqrt(Br**2+Bt**2+Bz**2)
        magnitude[magnitude == 0] = 1

        Br/=magnitude
        Bz/=magnitude
        Bt/=magnitude

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



def main():

    z=np.linspace(-2000,2000,25)
    r=np.linspace(0.1,1500,15)
    theta=np.linspace(0,2*np.pi,4)

    k=0.01
    zc=1000

    magnetic_field=MagneticField(k, zc,r,theta,z)

    R=magnetic_field.R
    Bx,By,Bz,X,Y,Z=magnetic_field.Compute_field()
    Br=magnetic_field.Br
    gradient_magnitude=magnetic_field.Compute_gradient()

    fig=plt.figure(figsize=(40,40))
    ax=fig.add_subplot(111, projection='3d')
    ax.quiver(X,Y,Z,Bx,By,Bz,color='black',label='Magnetic Field Vectors',length=100,linewidth=2,arrow_length_ratio=0.5,normalize=True)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    # plt.show()


    plt.figure()
    plt.pcolormesh(Z[:,0,:],R[:,0,:],gradient_magnitude[:,0,:],cmap='viridis')
    plt.colorbar(label='|grad(B)|')
    plt.quiver(Z[:,0,:],R[:,0,:],Bz[:,0,:],Br[:,0,:],color='white',label='B',width=0.003,scale=50,headwidth=3)
    plt.xlabel('z')
    plt.ylabel('r')
    plt.title('Magnetic Field and its gradient in Magnetic Mirror Configuration')
    plt.legend()
    # plt.savefig('B_field_mirror.png')
    plt.show()





if __name__ == "__main__":
    main()