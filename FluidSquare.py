"""
Fluid Simulation based on:
https://thecodingtrain.com/CodingChallenges/132-fluid-simulation.html
https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

MIT License
"""
from enum import Enum
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
import noise
from random import randint

plt.style.use('dark_background')


class EdgeUpdateType(Enum):
    DENSITY = 0
    X_VELOCITY = 1
    Y_VELOCITY = 2


def index(x, y, n):
    return x + y * n


class FluidSquare:

    def __init__(self, diffusion, viscosity, dt, size=32, cell_size=4):
        """
        Constructor
        :param diffusion: Diffusion amount of how velocities and dyes spread
        :param viscosity: Viscosity amount
        :param dt: Time delta
        :param size: Size of fluid square for x and y dimensions, individually
        :param cell_size: The size of each fluid cell
        """
        self.size = size
        self.t = 0
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity
        # densities of dye
        self.densities = np.zeros(shape=(size * size), dtype=float)
        self.previous_densities = np.zeros(shape=(size * size), dtype=float)
        self.x_velocities = np.zeros(shape=(size * size), dtype=float)
        self.y_velocities = np.zeros(shape=(size * size), dtype=float)
        self.previous_x_velocities = np.zeros(shape=(size * size), dtype=float)
        self.previous_y_velocities = np.zeros(shape=(size * size), dtype=float)
        self.cell_size = cell_size
        self.fig = plt.figure(figsize=(3, 3), dpi=100)
        self.ax = plt.axes(xlim=(0, size * cell_size), ylim=(0, size * cell_size))
        self.density_grid_cells = np.full((size * size), patches.Rectangle((0, 0), cell_size, cell_size))

        for y in range(0, self.size):
            for x in range(0, self.size):
                self.density_grid_cells[index(x, y, size)] = patches.Rectangle((x * cell_size, y * cell_size),
                                                                               cell_size, cell_size)
                self.ax.add_patch(self.density_grid_cells[index(x, y, size)])

    def _step(self):
        """
        Completes one time step of the simulation
        :return:
        """
        #x_velocities = self.x_velocities
        #y_velocities = self.y_velocities
        #previous_x_velocities = self.previous_x_velocities
        #previous_y_velocities = self.previous_y_velocities
        #densities = self.densities
        #previous_densities = self.previous_densities

        FluidSquare._diffuse(EdgeUpdateType.X_VELOCITY, self.previous_x_velocities, self.x_velocities, self.viscosity, self.dt)
        FluidSquare._diffuse(EdgeUpdateType.Y_VELOCITY, self.previous_y_velocities, self.y_velocities, self.viscosity, self.dt)

        FluidSquare._project(self.previous_x_velocities, self.previous_y_velocities, self.x_velocities, self.y_velocities)

        FluidSquare._advect(EdgeUpdateType.X_VELOCITY, self.x_velocities, self.previous_x_velocities, self.previous_x_velocities,
                            self.previous_y_velocities, self.dt)

        FluidSquare._advect(EdgeUpdateType.Y_VELOCITY, self.y_velocities, self.previous_y_velocities, self.previous_x_velocities,
                            self.previous_y_velocities, self.dt)

        FluidSquare._project(self.x_velocities, self.y_velocities, self.previous_x_velocities, self.previous_y_velocities)
        FluidSquare._diffuse(EdgeUpdateType.DENSITY, self.previous_densities, self.densities, self.diffusion, self.dt)

        FluidSquare._advect(EdgeUpdateType.DENSITY, self.densities, self.previous_densities, self.x_velocities, self.y_velocities,
                            self.dt)

    def _render_densities(self):
        """
        Render density cells
        :return:
        """
        max_density = np.max(self.densities)
        n = int(np.sqrt(len(self.densities)))

        for y in range(0, n):
            for x in range(0, n):
                # set cell colors by densities
                density_color = self.densities[index(x, y, n)] / max_density
                self.density_grid_cells[index(x, y, n)].set_color([density_color, density_color, density_color])

    @staticmethod
    def _diffuse(edge_update_type, input_vector, previous_input_vector, diffusion, dt, iterations=4):
        """
        Makes dye or velocities spread out
        :param edge_update_type: The type of edge update to do after diffusion
        :param input_vector: Density or velocity matrix
        :param previous_input_vector: Previous density or velocity matrix
        :param diffusion: Amount of diffusion
        :param dt: Time delta
        :param iterations: Number of iterations to use in linear solve
        """
        n = np.sqrt(len(input_vector))
        a = dt * diffusion * (n - 2) * (n - 2)
        c = 1 + 6 * a
        FluidSquare._linear_solve(edge_update_type, input_vector, previous_input_vector, a, c, iterations)

    @staticmethod
    def _update_boundary_cells(edge_update_type, input_vector):
        """
        Updates boundary cell densities or velocities.  The edges are mirrored to simulate fluids not leaking
        :param edge_update_type: The type of edge update to do
        :param input_matrix: The input densities or velocities
        :return:
        """
        n = int(np.sqrt(len(input_vector)))

        for x in range(1, n - 1):
            # fill top row
            input_vector[index(x, 0, n)] = -input_vector[index(x, 1, n)] if edge_update_type is EdgeUpdateType.Y_VELOCITY else input_vector[index(x, 1, n)]

            # fill bottom row
            input_vector[index(x, n - 1, n)] = -input_vector[index(x, n - 2, n)] if edge_update_type is EdgeUpdateType.Y_VELOCITY else input_vector[index(x, n - 2, n)]

        for y in range(1, n - 1):
            # fill left-most column
            input_vector[index(0, y, n)] = -input_vector[index(1, y, n)] if edge_update_type is EdgeUpdateType.X_VELOCITY else input_vector[index(1, y, n)]

            input_vector[index(n - 1, y, n)] = -input_vector[index(n - 2, y, n)] if edge_update_type is EdgeUpdateType.X_VELOCITY else input_vector[index(n - 2, y, n)]

        # set upper-left corner
        input_vector[index(0, 0, n)] = .5 * (input_vector[index(1, 0, n)] + input_vector[index(0, 1, n)])

        # set bottom-left corner
        input_vector[index(0, n - 1, n)] = .5 * (input_vector[index(1, n - 1, n)] + input_vector[index(0, n - 2, n)])

        # set top-right corner
        input_vector[index(n - 1, 0, n)] = .5 * (input_vector[index(n - 2, 0, n)] + input_vector[index(1, n - 1, n)])

        # set bottom-right corner
        input_vector[index(n - 1, n - 1, n)] = .5 * (input_vector[index(n - 2, n - 1, n)] + input_vector[index(n - 1, n - 2, n)])

    @staticmethod
    def _linear_solve(edge_update_type, input_vector, previous_input_vector, a, c, iterations):
        c_reciprocal = 1. / c
        n = int(np.sqrt(len(input_vector)))

        for i in range(0, iterations):
            for y in range(1, n - 1):
                for x in range(1, n - 1):
                    input_vector[index(x, y, n)] = (previous_input_vector[index(x, y, n)] + a *
                                                    (input_vector[index(x + 1, y, n)] +
                                                     input_vector[index(x - 1, y, n)] +
                                                     input_vector[index(x, y + 1, n)] +
                                                     input_vector[index(x, y - 1, n)])) * c_reciprocal

            FluidSquare._update_boundary_cells(edge_update_type, input_vector)

    @staticmethod
    def _project(x_velocities, y_velocities, p, div, iterations=4):
        """
        Runs through all simulation cells and makes sure they're at equilibrium
        :param x_velocities: The cell velocities in the x direction
        :param y_velocities: The cell velocities in the y direction
        :param p:
        :param div:
        :param iterations: Number of iterations to use in linear solve
        """
        n = int(np.sqrt(len(x_velocities)))

        for y in range(1, n - 1):
            for x in range(1, n - 1):
                div[index(x, y, n)] = (-.5 * (
                    x_velocities[index(x + 1, y, n)] -
                    x_velocities[index(x - 1, y, n)] +
                    y_velocities[index(x, y + 1, n)] -
                    y_velocities[index(x, y - 1, n)])) / n

                p[index(x, y, n)] = 0

        FluidSquare._update_boundary_cells(EdgeUpdateType.DENSITY, div)
        FluidSquare._update_boundary_cells(EdgeUpdateType.DENSITY, p)
        FluidSquare._linear_solve(EdgeUpdateType.DENSITY, p, div, 1, 6, iterations)

        for y in range(1, n - 1):
            for x in range(1, n - 1):
                x_velocities[index(x, y, n)] -= .5 * (p[index(x + 1, y, n)] - p[index(x - 1, y, n)]) * n
                y_velocities[index(x, y, n)] -= .5 * (p[index(x, y + 1, n)] - p[index(x, y - 1, n)]) * n

        FluidSquare._update_boundary_cells(EdgeUpdateType.X_VELOCITY, x_velocities)
        FluidSquare._update_boundary_cells(EdgeUpdateType.Y_VELOCITY, y_velocities)

    @staticmethod
    def _advect(edge_update_type, d, previous_d, x_velocities, y_velocities, dt):
        n = int(np.sqrt(len(d)))
        dt_x = dt * (n - 2)

        for j in range(1, n - 1):  # y
            for i in range(1, n - 1):  # x
                temp1 = dt_x * x_velocities[index(i, j, n)]
                temp2 = dt_x * y_velocities[index(i, j, n)]
                x = i - temp1
                y = j - temp2

                if x < .5:
                    x = .5

                if x > n + .5:
                    x = n + .5

                i0 = np.floor(x)
                i1 = i0 + 1.

                if y < .5:
                    y = .5

                if y > n + .5:
                    y = n + .5

                j0 = np.floor(y)
                j1 = j0 + 1.

                s1 = x - i0
                s0 = 1. - s1
                t1 = y - j0
                t0 = 1. - t1

                i0 = min(n - 1, int(i0))
                i1 = min(n - 1, int(i1))
                j0 = min(n - 1, int(j0))
                j1 = min(n - 1, int(j1))

                d[index(i, j, n)] = s0 * (t0 * previous_d[index(i0, j0, n)] + t1 * previous_d[index(i0, j1, n)]) + \
                                    s1 * (t0 * previous_d[index(i1, j0, n)] + t1 * previous_d[index(i1, j1, n)])

        FluidSquare._update_boundary_cells(edge_update_type, d)

    def _draw(self, frame):
        """
        Draws one animation frame
        :param frame: Frame number
        :return:
        """
        center_cell = int(.5 * self.size)
        n = self.size

        # add density
        for x in range(-1, 2):
            for y in range(-1, 2):
                self.densities[index(center_cell + x, center_cell + y, n)] += randint(50, 150)

        # add velocity
        for i in range(0, 2):
            angle = 2 * np.pi * noise.pnoise1(self.t)
            v = np.array([np.cos(angle), np.sin(angle)])
            v *= .2
            self.t += .1
            self.x_velocities[index(center_cell, center_cell, n)] += v[0]
            self.y_velocities[index(center_cell, center_cell, n)] += v[1]

        self._step()
        self._render_densities()

    def run(self):
        plt.axis('off')
        a = animation.FuncAnimation(self.fig, self._draw, interval=1)
        # a.save('animation.gif', writer='imagemagick')
        plt.show()

