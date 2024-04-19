import matplotlib.pyplot as plt
import numpy as np
import math

def rotate_point(point, center, angle):
    """Rotate a point around a center by a given angle."""
    ox, oy = center
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return qx, qy

def ellipse_to_triangles(ax, center, width, height, num_triangles=8, angle = 0):
    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, num_triangles+1)
    x = center[0] + 0.5 * width * np.cos(theta)
    y = center[1] + 0.5 * height * np.sin(theta)

    # Plot the ellipse
    #ax.plot(x, y)
    rotated_points = [rotate_point((x[i], y[i]), center, angle) for i in range(len(x))]

    # Approximate the ellipse with triangles
    for i in range(num_triangles):
        triangle_points = np.array([
            [center[0], center[1]],
            rotated_points[i],
            rotated_points[i+1],
            [center[0], center[1]]
        ])
        # Print triangle coordinates
        triangle_coordinates = triangle_points.flatten()[0:6]
        print(f"{{{','.join(map(str, triangle_coordinates))}}}")
        # Plot each triangle
        ax.plot(triangle_points[:, 0], triangle_points[:, 1], linestyle='solid', color ='black')

def main():
    centerx = [0,0,0.22,-0.22,0,0,0,-0.08,0,0.06]
    centery = [0,-0.0184,0,0,0.35,0.1,-0.1,-0.605,-0.605,-0.605]
    awidth = [0.69,0.6624,0.11,0.16,0.21,0.046,0.046,0.046,0.023,0.023]
    aheight = [0.92,0.874,0.31,0.41,0.25,0.046,0.046,0.023,0.023,0.046]
    #18 degrees = 0.314159 radians
    astart_theta = [0,0,-0.314159,0.314159,0,0,0,0,0,0]
    # Create a plot
    fig, ax = plt.subplots()    
    
    # Input ellipse parameters
    for i in range(0,len(centerx)):
        center = (centerx[i], centery[i])
        width = awidth[i] * 2
        height = aheight[i] * 2
        polygon_num = int(np.floor(50 * math.sqrt(0.5 * (width * height + 0.1))))
        #print("ellipse: {}  polynum: {}".format(i,polygon_num))
        # Convert ellipse to triangles and plot
        ellipse_to_triangles(ax, center, width, height, num_triangles=polygon_num, angle=astart_theta[i])

    # Set plot properties
    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.title('Ellipse to Triangles Approximation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
