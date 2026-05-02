# ode_solver.py
#import the necessary libraries for the ODE solver implementation

import jax 
import jax.numpy as jnp
from functools import partial 
from typing import Callable, Tuple

# 1. Euler method implementation

@partial(jax.jit, static_argnums=(0,))
def euler_step(f: Callable, y: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:
    """Single Euler step: y_{n+1} = y_n + dt * f(t_n, y_n)"""
    return y + dt * f(t, y)

@partial(jax.jit, static_argnums=(0,))
def euler_solve(
    f: Callable,
    y0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve ODE using Euler method.
    
    Args:
        f:   dy/dt = f(t, y)
        y0:  initial condition
        t0:  start time
        t1:  end time
        dt:  step size

    Returns:
        (times, ys) arrays
    """
    times = jnp.arange(t0, t1, dt)

    def step(y, t):
        y_next = euler_step(f, y, t, dt)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, times)
    return times, ys


# 2. RK4 method implementation

@partial(jax.jit, static_argnums=(0,))
def rk4_step(f: Callable, y: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:
    """Single RK4 step."""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

@partial(jax.jit, static_argnums=(0,))
def rk4_solve(
    f: Callable,
    y0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve ODE using RK4 method.
    
    Args:
        f:   dy/dt = f(t, y)
        y0:  initial condition
        t0:  start time
        t1:  end time
        dt:  step size

    Returns:
        (times, ys) arrays
    """
    times = jnp.arange(t0, t1, dt)

    def step(y, t):
        y_next = rk4_step(f, y, t, dt)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, times)
    return times, ys

# 3. Batch solving with vmap (jit vs non-jit)

def batch_solve(
    solver_fn: Callable,
    f: Callable,
    y0_batch: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
):
    """
    Solve the same ODE for multiple initial conditions using vmap.

    Args:
        solver_fn:    euler_solve or rk4_solve
        y0_batch:  batch of initial conditions (batch_size, state_dim)

    Returns:
        (times, ys) where ys has shape (batch_size, steps, state_dim)
    """
    vmapped = jax.vmap(
        lambda y0: solver_fn(f, y0, t0, t1, dt)
    )
    return vmapped(y0_batch)

