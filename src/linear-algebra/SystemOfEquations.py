from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt


class FoSysEq:

    def __init__(self, a: np.array, b: np.array):
        """
        Initializes the FoSysEq object with the provided matrices a and b.

        Solves for x in the form of the linear matrix equation ax = b,
        where a is a 2x2 or 3x3 matrix and x and b are 2x1 or 3x1 vectors
        and can be used to plot the solution.

        For example:
        Where a is a 3x3 matrix and b is a 3x1 vector.

        The three equations:
        eq0: 1x + 3y - 2z =  2
        eq1: 2x + 1y + 4z = -1
        eq2: 3x - 2y +  z =  1

        Can be written in linear matrix equation form, ax = b, as follows:
              |1,  3, -2|      |x|      | 2|
        a =   |2,  1,  4|, x = |y|, b = |-1|
              |3, -2,  1|      |z|      | 1|

        Parameters
        ----------
        a : np.array
            The matrix a representing the x, y, and z coefficients of the system of equations.
        b : np.array
            The matrix b representing the constant values of the system of equations.

        Attributes
        ----------
        x : np.array
            The solution of the system of equations.

        Returns
        -------
        None

        Raises
        ------
        np.linalg.LinAlgError
            If the matrix a is not square or if the matrix a is not 2x2 or 3x3.
        """

        m, n = a.shape[-2:]
        if m != n:
            raise np.linalg.LinAlgError('The matrix a must be square.')

        if m > 3:
            raise np.linalg.LinAlgError('The matrix a must be 2x2 or 3x3.')

        self.a: np.array = a
        self.b: np.array = b
        self.x: np.array = self.solve()
        self.xlim = (self.x[0]-2, self.x[0]+2)
        self.ylim = (self.x[1]-2, self.x[1]+2)
        self.colors: list[str] = ['r', 'g', 'b']

        self.is_3d = False
        self.x_range: np.ndarray[np.Any, np.dtype[np.floating[np.Any]]] = np.arange(-5, 5, 0.25)
        if a.shape[0] == 3:
            self.is_3d = True
            self.zlim = (self.x[2]-1, self.x[2]+1)
            self.surf_mat = self.calc_3d_surface_equations_matix()
            self.line_mat = self.calc_3d_line_intersection_matrix()
            self.y_range: np.ndarray[np.Any, np.dtype[np.floating[np.Any]]] = np.arange(-5, 5, 0.25)
        else:
            self.line_mat = self.calc_2d_line_equations_matix()

    def solve(self) -> np.array:
        return np.linalg.solve(self.a, self.b)

    def lstsq(self) -> np.array:
        return np.linalg.lstsq(self.a, self.b, rcond=None)

    def calc_2d_line_equations_matix(self) -> np.array:
        a_line = self.a / (self.a[:, 1][:, np.newaxis])*-1
        b_line = self.b / self.a[:, 1]
        return np.insert(a_line[:, :1], 1, b_line, axis=1)

    def line_equation(self, x: np.array, idx: int) -> np.array:
        return x*self.line_mat[idx, 0] + self.line_mat[idx, 1]

    def calc_3d_surface_equations_matix(self) -> np.array:
        """
        Defines a matrix of the three equations for plotting the 3D surfaces.
        Solve for z by dividing each element in a and b by z coefficients
        and then negate a.

        Returns
        -------
        A 3x3 np.array of the three 3D surface equations where:
        z = ax + by + c

        For example:
        Where a is a 3x3 matrix and b is a 3x1 vector.
        a:
        |1,  3, -2|
        |2,  1,  4|
        |3, -2,  1|

        b:
        | 2|
        |-1|
        | 1|

        eq0: z =  1/2x + 3/2y - 1
        eq1: z = -1/2x - 1/4y - 1/4
        eq2: z =   -3x +   2y + 1

        Result:
        [[ 0.5   1.5  -1.  ]
         [-0.5  -0.25 -0.25]
         [-3.    2.    1.  ]]
        """
        a_surface = self.a / (self.a[:, 2][:, np.newaxis])*-1
        b_surface = self.b / self.a[:, 2]
        return np.insert(a_surface[:, :2], 2, b_surface, axis=1)

    def surface_equation(self, x: np.array, y: np.array, idx: int) -> np.array:
        return x*self.surf_mat[idx, 0] + y*self.surf_mat[idx, 1] + self.surf_mat[idx, 2]

    def calc_3d_line_intersection_matrix(self) -> np.array:
        line_matrix = np.empty((3, 2))
        for i in range(3):
            next_i = (i + 1) % 3
            multiplier = self.a[next_i][2]/self.a[i][2]*-1
            a_line = np.array(self.a[i] * multiplier) + np.array(self.a[next_i])
            b_line = np.array(self.b[i] * multiplier) + np.array(self.b[next_i])
            line_matrix[i] = np.array([a_line[0]/a_line[1]*-1, b_line/a_line[1]])
        return line_matrix

    def plot_3d_solution(self) -> tuple[Figure, Axes]:
        # Prepare 3D figure and label axes
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(self.x_range, self.y_range)

        # Plot 3D surfaces and intersection lines
        for i in range(3):
            ax.plot_surface(X, Y, self.surface_equation(X, Y, i), color=self.colors[i], alpha=0.2, label=f'eq{i}')
            # Plot the line intersections
            y = self.line_equation(self.x_range, i)
            ax.plot(self.x_range, y, self.surface_equation(self.x_range, y, i), color=self.colors[i])

        # Plot 3D solution point
        ax.plot(*self.x, 'o', markersize=5, color='k', label='Solution')

        return fig, ax

    def plot_2d_solution(self) -> tuple[Figure, Axes]:
        # Prepare 2D figure and label axes
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)

        # Plot 3D surfaces and intersection lines
        for i in range(2):
            ax.plot(self.x_range, self.line_equation(self.x_range, i), color=self.colors[i], label=f'eq{i}')

        # Plot 2D solution point
        plt.plot(*self.x, 'o', markersize=5, color='k', label='Solution')

        return fig, ax

    def plot_solution(self) -> tuple[Figure, Axes]:
        if self.is_3d:
            return self.plot_3d_solution()

        return self.plot_2d_solution()
