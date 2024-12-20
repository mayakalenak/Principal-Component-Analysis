'''scatter3d.py
3D scatterplot to help visualize 3D numeric data
Oliver W. Layton
CS 251: Data Analysis and Visualization
Fall 2024
'''
import matplotlib.pyplot as plt

def scatter3dplot(data_xyz, color=None, headers=['x', 'y', 'z'],
                  title='Mystery dataset', lims=(0, 1)):
    '''Creates a 3D scatter plot to visualize data'''
    curr_ax = plt.axes(projection='3d')

    # Scatter plot of data in 3D
    if color is None:
        curr_ax.scatter3D(data_xyz[:, 0], data_xyz[:, 1], data_xyz[:, 2],
                     cmap='viridis')
    else:
        curr_ax.scatter3D(data_xyz[:, 0], data_xyz[:, 1], data_xyz[:, 2],
                     c=color, cmap='viridis')

    # style the plot
    curr_ax.set_xlabel(headers[0])
    curr_ax.set_ylabel(headers[1])
    curr_ax.set_zlabel(headers[2])
    curr_ax.set_title(title)

    curr_ax.set_xlim(lims[0], lims[1])
    curr_ax.set_ylim(lims[0], lims[1])
    curr_ax.set_zlim(lims[0], lims[1])

    plt.show()
