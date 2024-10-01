import matplotlib.pyplot as plt
import numpy as np

class MPCC_plotter:
    def __init__(self):

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

        # Initialize a line object, that will be updated
        self.line, = self.ax.plot([], [], 'ro')  # 'ro' means red color, circle markers

        self.ref_point, = self.ax.plot([], [], 'ko')
        self.ref_line, = self.ax.plot([],[], 'b')


        # Data container
        self.data = {'x': [], 'y': []}

    def set_ref_point(self, x,y):
        self.ref_point.set_data([x],[y])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_ref_traj(self, x,y):
        self.ref_line.set_data(x, y)
    
    def update_plot(self, new_x, new_y):
        """Update the plot with new point, removing the previous one."""
        # Clear previous points
        self.data['x'].clear()
        self.data['y'].clear()

        # Add new point
        self.data['x'].append(new_x)
        self.data['y'].append(new_y)

        # Update line data
        self.line.set_data(self.data['x'], self.data['y'])

        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # Show the plot in interactive mode

    def show(self):
        plt.ion()
        plt.show()
