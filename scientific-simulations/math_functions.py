import numpy as np
import random

def polynomial_function_array(coefficients, exponents, x_points):
  y_points = []
  for x in x_points:
      y = np.sum(np.array(coefficients) * x**np.array(exponents))
      y_points.append(y)
  return np.array(y_points)


def polynomial_function_number(coefficients, exponents, x):
  y = np.sum(np.array(coefficients) * x**np.array(exponents))
  return y
  

def noise_generator(error):
  noise = random.random()
  while noise > error:
    noise = random.random()
  return noise


def exponential_decay(t, N_0, tau):
  N = N_0 * np.exp(-t/tau)
  return N


def euler_ODE_appr(y_0, t_begin, t_end, N, f_diff, factors):
  h = (t_end - t_begin)/N
  t_points = np.arange(t_begin, t_end, h)
  y_points = []
  y_points.append(y_0)
  for i in range(N-1):
    y_points.append(y_points[i] + h*f_diff(y_points[i], factors))
  return t_points, y_points


def random_walk(p_0, N):
  step_array = np.arange(0, N+1, 1)
  position_array = np.zeros((N+1, 2))
  position_array[[0], :] = p_0
  for step in range(1, N+1):
    n = random.choice([-1, 1])
    if n < 0:
      position_array[step] = position_array[step-1] + np.array([random.choice([-1, 1]), 0])
    else:
      position_array[step] = position_array[step-1] + np.array([0, random.choice([-1, 1])])
  return step_array, position_array


