import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cbook
from matplotlib.colors import LightSource

# Load and format data
dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
z = dem['elevation']
nrows, ncols = z.shape
x = np.linspace(dem['xmin'], dem['xmax'], ncols)
y = np.linspace(dem['ymin'], dem['ymax'], nrows)


x, y = np.meshgrid(x, y)

region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

#-------------- finding random point-----------#
len_z=np.size(z)
idx_z=np.arange(0 , len_z)
print(f"len z = {len_z}")
print(f"index z =\n {idx_z}")

random_idx= np.random.choice(idx_z)
print(f"random point = {random_idx}")

point=z.flat[random_idx]
print(f"point = {point}")

idx_xy= np.where(z==point)
print(idx_xy)

row = idx_xy[0][0]
col= idx_xy[1][0]
print(f"row : {row}")
print(f"col : {col}")

x_point=x[row , col]
y_point=y[row , col]
z_point=z[row , col]
print(f"x_point: {x_point}")
print(f"y_point: {y_point}")
print(f"z_point: {z_point}")
#-----------------------------------------#
#-----------hyperparameter----------------#
lr=0.001
epoch= 2000
#------------------------------------------#
#------------ gradient descent-------------#
path_x = []
path_y = []
path_z = []

dz_dx = np.gradient(z)[1]
dz_dy = np.gradient(z)[0]

for i in range(epoch):

    #----------Gradient at current point---------#
    grad_x = dz_dx[int(row), int(col)]
    grad_y = dz_dy[int(row), int(col)]

    #---------Downward movement--------------#
    new_row = row - lr * grad_y
    new_col = col - lr * grad_x

    #-------Prevent out-of-bounds access---------#
    new_row = np.clip(new_row, 0, z.shape[0] - 1)
    new_col = np.clip(new_col, 0, z.shape[1] - 1)

    #-------Update point--------#
    row = new_row
    col = new_col

    x_point = x[int(row), int(col)]
    y_point = y[int(row), int(col)]
    z_point = z[int(row), int(col)]
    
    path_x.append(x_point)
    path_y.append(y_point)
    path_z.append(z_point)

    print(f"Iteration {i}: z = {z_point}")
    

#----------Set up plot----------#
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=plt.colormaps["gist_earth"], vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

# Draw the movement path on the 3D surface
ax.plot(path_x, path_y, path_z, color='red', linewidth=2)

# Starting point
ax.scatter(path_x[0], path_y[0], path_z[0],
           color='black', s=50, label='Start')

# Final point
ax.scatter(path_x[-1], path_y[-1], path_z[-1],
           color='red', s=50, label='End')

ax.legend()

plt.show()