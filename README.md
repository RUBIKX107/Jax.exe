# Jax ODE math solver

## Overview
Implement a numerical ODE solver in JAX supporting multiple methods,
demonstrating JIT compilation, vmap for batch solving, and gradient 
flow through the solver.

## Background
An ODE has the form:  dy/dt = f(t, y),  y(t0) = y0

We want to numerically integrate this forward in time using JAX primitives.

## Methods to Implement
- [ ] Euler method (1st order)
- [ ] RK4 — Runge-Kutta 4th order (gold standard)
- [ ] Benchmarking JIT vs non-JIT
- [ ] vmap over initial conditions (batch solving)

## Example Use Cases
- Simple harmonic oscillator: dy/dt = -ky
- Lotka-Volterra (predator-prey): coupled ODEs
- Exponential decay

## Deliverables
- `ode_solver.py` — core solver implementations
- `examples.py` — demo with plots
- Clean docstrings and type hints