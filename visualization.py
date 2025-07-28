import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D proection works
from quadcopter import Quadcopter

def set_data(obj, start=None, end=None, x_list=None, y_list=None, z_list=None):

    if x_list is None:
        obj.set_data([start[0], end[0]], [start[1], end[1]])
        obj.set_3d_properties([start[2], end[2]])
    else:
        obj.set_data(x_list, y_list)
        obj.set_3d_properties(z_list)

    return obj

class QuadcopterAnimator:
    def __init__(self, system):
        
        print("Animating quadcopter flight...")
        self.system = system

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.body = [self.ax.plot([], [], [], 'o-', lw=4, color='blue'),
                      self.ax.plot([], [], [], 'o-', lw=4, color='blue')]
        
        self.body_frame = [self.ax.plot([], [], [], '-', lw=2, color='black'),
                           self.ax.plot([], [], [], '-', lw=2, color='black'),
                           self.ax.plot([], [], [], '-', lw=2, color='black')]
        
        self.trajectory, = self.ax.plot([], [], [], '-', lw=1, color='green')
        self.trajectory_data = [[],[],[]]

        # Unpack the body plot objects
        self.body[0] = self.body[0][0]
        self.body[1] = self.body[1][0]
        self.body_frame[0] = self.body_frame[0][0]
        self.body_frame[1] = self.body_frame[1][0]
        self.body_frame[2] = self.body_frame[2][0]

        self.body_axis_length = 0.2
        self.body_length = 0.5
        self.speed = 4  # Speed multiplier for the animation
        grid_resolution = 5

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xticks(np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], grid_resolution))
        self.ax.set_yticks(np.linspace(self.ax.get_ylim()[0], self.ax.get_ylim()[1], grid_resolution))
        self.ax.set_zticks(np.linspace(self.ax.get_zlim()[0], self.ax.get_zlim()[1], grid_resolution))
        self.ax.set_title(f"Quadcopter Animation, {self.speed}x speed")
        self.ax.set_aspect('equal')
        self.ax.view_init(elev=25, azim=30, roll=0)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
    def update_frame(self, frame):
        x = self.x[:, frame]
        
        phi, theta, psi = x[0:3]
        R_B_to_B0 = self.system.B_to_B0_matrix(phi, theta, psi)
        R_B_to_S = self.system.B0_to_S(R_B_to_B0)

        pos_S = self.system.B0_to_S(x[9:12]) # Position of drone in world frame S

        # Draw trajectory
        self.trajectory_data[0].append(pos_S[0])
        self.trajectory_data[1].append(pos_S[1])
        self.trajectory_data[2].append(pos_S[2])
        self.trajectory = set_data(obj=self.trajectory, 
                                   x_list=self.trajectory_data[0],
                                   y_list=self.trajectory_data[1],
                                   z_list=self.trajectory_data[2])

        x_Bhat_in_S = R_B_to_S[:,0]
        y_Bhat_in_S = R_B_to_S[:,1]
        z_Bhat_in_S = R_B_to_S[:,2]

        # Update the body frame lines
        self.body_frame[0] = set_data(obj=self.body_frame[0], start=pos_S, end=pos_S + self.body_axis_length * x_Bhat_in_S)
        self.body_frame[1] = set_data(obj=self.body_frame[1], start=pos_S, end=pos_S + self.body_axis_length * y_Bhat_in_S)
        self.body_frame[2] = set_data(obj=self.body_frame[2], start=pos_S, end=pos_S + self.body_axis_length * z_Bhat_in_S)

        # Update the body
        R_45_to_B = np.array([[np.cos(np.pi/4), np.sin(np.pi/4), 0],
                              [-np.sin(np.pi/4), np.cos(np.pi/4), 0],
                              [0, 0, 1]])
        R_45_to_S = R_B_to_S @ R_45_to_B
        x_45_in_S = R_45_to_S[:, 0]
        y_45_in_S = R_45_to_S[:, 1]

        self.body[0] = set_data(obj=self.body[0], start=pos_S - self.body_length/2 * x_45_in_S, end=pos_S + self.body_length/2 * x_45_in_S)
        self.body[1] = set_data(obj=self.body[1], start=pos_S - self.body_length/2 * y_45_in_S, end=pos_S + self.body_length/2 * y_45_in_S)

        artist_list = [self.body[0], self.body[1],
                       self.body_frame[0], self.body_frame[1], self.body_frame[2],
                       self.trajectory]

        return artist_list
    
    def init_frame(self):
        self.body[0] = set_data(self.body[0], [], [], [])
        self.body[1] = set_data(self.body[1], [], [], [])

        self.body_frame[0] = set_data(self.body_frame[0], [], [], [])
        self.body_frame[1] = set_data(self.body_frame[1], [], [], [])
        self.body_frame[2] = set_data(self.body_frame[2], [], [], [])

        self.trajectory = set_data(self.trajectory, [], [], [])

        artist_list = [self.body[0], self.body[1],
                       self.body_frame[0], self.body_frame[1], self.body_frame[2],
                       self.trajectory]

        return artist_list
    
    def animate(self, data, output_filename="deepc_quadcopter_animation.mp4", fps=30):
        self.time = data["time"]
        self.x = data["x"]  # State

        # Use the fixed time step
        dt = self.time[1] - self.time[0]  # Fixed time step
        interval = 1000 / fps  # Interval in milliseconds for FuncAnimation

        # Subsample the data to match desired FPSinterval
        skip = self.speed * int(1 / (dt * fps))
        self.time = self.time[::skip].copy()
        self.x = self.x[:, ::skip].copy()

        # Create the animation object with a fixed interval
        ani = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=len(self.time),
            init_func=self.init_frame,
            blit=True,
            interval=interval
        )

        # Setting up FFMpegWriter to save the animation
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={'artist': 'Me'}, bitrate=1800)

        # Save the animation as an MP4 file
        ani.save(f"DeePC_Quadcopter/Movies/{output_filename}", writer=writer)

        print(f"Animation saved as {output_filename} in DeePC_Quadcopter/Movies")