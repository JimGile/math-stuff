import numpy as np
import matplotlib.pyplot as plt

# Using NumPy linalg.solve
# Computes the "exact" solution, x, of the well-determined, full rank, linear matrix equation ax = b.
# Solve the system of equations x0 + 2 * x1 = 1 and 3 * x0 + 5 * x1 = 2:
a = np.array([[1, 2], [3, 5]])
b = np.array([1, 2])
x = np.linalg.solve(a, b)
print(x)

# Check that the solution is correct:
print(np.allclose(np.dot(a, x), b))

# solves the system of linear equations:
# 3x + 2y = 9
# 2x + 4y = 8
a = np.array([[3, 2], [2, 4]])
b = np.array([9, 8])
x = np.linalg.solve(a, b)
print(x)

# Check that the solution is correct:
print(np.allclose(np.dot(a, x), b))

# Using NumPy linalg.lstsq
# Solve the system of linear equations:
# 3x + 2y = 9
# 2x + 4y = 8
a = np.array([[3, 2], [2, 4]])
b = np.array([9, 8])
x, res, rnk, s = np.linalg.lstsq(a, b, rcond=None)
print(x)
print(res)
print(rnk)

# Check that the solution is correct:
print(np.allclose(np.dot(a, x), b))


# Fit a line, y = mx + c, through some noisy data-points:
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
# By examining the coefficients, we see that the line should have a gradient of roughly 1
# and cut the y-axis at, more or less, -1.
# We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]].
# Now use lstsq to solve for p:

A = np.vstack([x, np.ones(len(x))]).T
print(A)

# array([[ 0.,  1.],
#        [ 1.,  1.],
#        [ 2.,  1.],
#        [ 3.,  1.]])

m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, c)
# (1.0 -0.95) # may vary

# Plot the 2D data along with the fitted line:
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m*x + c, 'r', label='Fitted line')
_ = plt.legend()
plt.show()


# Solve the following system of linear equations
# using the linalg.solve function.
# 1x + 3y -2z = 2
# 2x + 1y + 4z = -1
# 3x - 2y + z = 1
#
# Solve for x the linear matrix equation ax = b.
# Where a is a 3x3 matrix and x and b are 3x1 vectors.
#     |1,  3, -2|      |x|      | 2|
# a = |2,  1,  4|, x = |y|, b = |-1|
#     |3, -2,  1|      |z|      | 1|
#
a: np.array = np.array([
    [1,  3, -2],
    [2,  1,  4],
    [3, -2,  1]
    ])

b: np.array = np.array([
    2,
    -1,
    1
    ])

x = np.linalg.solve(a, b)
print(x)

x_sol = np.array([31/53, 5/53, -30/53])
print(x_sol)
print(np.allclose(x, x_sol))


# Define the three equations
# eq0: z =  1/2x + 3/2y - 1
# eq1: z = -1/2x - 1/4y - 1/4
# eq2: z =   -3x +   2y + 1
def eq0(x, y):
    return 1/2*x + 3/2*y - 1


def eq1(x, y):
    return -1/2*x - 1/4*y - 1/4


def eq2(x, y):
    return -3*x + 2*y + 1


# Plot the three 3D surface equations along with a marker for the solution:
# Define the ranges for x and y
X = np.arange(-1, 2, 0.25)
Y = np.arange(-1, 2, 0.25)
X, Y = np.meshgrid(X, Y)

# Prepare 3D figure and label axes
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot the equations
ax.plot_surface(X, Y, eq0(X, Y), color='r', alpha=0.2, label='eq0')
ax.plot_surface(X, Y, eq1(X, Y), color='g', alpha=0.2, label='eq1')
ax.plot_surface(X, Y, eq2(X, Y), color='b', alpha=0.2, label='eq2')

# Plot the solution
# ax.plot(x[0], x[1], x[2], 'o', markersize=2, color='k')
ax.plot(*x, 'o', markersize=2, color='k', label='Solution')
ax.legend()

plt.show()
