# examples.py

import jax.numpy as jnp
import matplotlib.pyplot as plt
from ode_solver import euler_solve, rk4_solve, batch_solve

# ── Example 1: Exponential Decay ───────────────────────────────────
# dy/dt = -k*y  →  exact solution: y(t) = y0 * exp(-k*t)

k = 0.5

def exp_decay(t, y):
    return -k * y

y0    = jnp.array([1.0])
times_e, ys_e = euler_solve(exp_decay, y0, 0.0, 10.0, 0.1)
times_r, ys_r = rk4_solve(exp_decay,  y0, 0.0, 10.0, 0.1)
exact         = jnp.exp(-k * times_r)

plt.figure(figsize=(8, 4))
plt.plot(times_r, exact,        label="Exact",  linestyle="--")
plt.plot(times_e, ys_e[:, 0],   label="Euler")
plt.plot(times_r, ys_r[:, 0],   label="RK4")
plt.title("Exponential Decay")
plt.legend()
plt.show()


# ── Example 2: Simple Harmonic Oscillator ──────────────────────────
# State: [position, velocity]
# d(pos)/dt = vel
# d(vel)/dt = -omega^2 * pos

omega = 2.0

def harmonic_oscillator(t, y):
    pos, vel = y
    return jnp.array([vel, -omega**2 * pos])

y0 = jnp.array([1.0, 0.0])   # start at x=1, v=0
times, ys = rk4_solve(harmonic_oscillator, y0, 0.0, 10.0, 0.05)

plt.figure(figsize=(8, 4))
plt.plot(times, ys[:, 0], label="Position")
plt.plot(times, ys[:, 1], label="Velocity")
plt.title("Simple Harmonic Oscillator (RK4)")
plt.legend()
plt.show()


# ── Example 3: Lotka-Volterra (Predator-Prey) ──────────────────────
# dx/dt = alpha*x - beta*x*y     (prey)
# dy/dt = delta*x*y - gamma*y    (predator)

alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5

def lotka_volterra(t, y):
    prey, pred = y
    dprey = alpha * prey  - beta  * prey * pred
    dpred = delta * prey * pred - gamma * pred
    return jnp.array([dprey, dpred])

y0 = jnp.array([10.0, 5.0])
times, ys = rk4_solve(lotka_volterra, y0, 0.0, 30.0, 0.01)

plt.figure(figsize=(8, 4))
plt.plot(times, ys[:, 0], label="Prey")
plt.plot(times, ys[:, 1], label="Predator")
plt.title("Lotka-Volterra (RK4)")
plt.legend()
plt.show()


# ── Example 4: Batch Solving ───────────────────────────────────────
# Solve harmonic oscillator for many initial conditions at once

import jax.numpy as jnp

y0_batch = jnp.array([[a, 0.0] for a in [0.5, 1.0, 1.5, 2.0]])
times, ys_batch = batch_solve(rk4_solve, harmonic_oscillator,
                               y0_batch, 0.0, 10.0, 0.05)

plt.figure(figsize=(8, 4))
for i, amp in enumerate([0.5, 1.0, 1.5, 2.0]):
    plt.plot(times, ys_batch[i, :, 0], label=f"x0={amp}")
plt.title("Batch Harmonic Oscillator (vmap + RK4)")
plt.legend()
plt.show()