import jax
import jax.numpy as jnp
import random

# ----------------------------
# Initialize parameters
# ----------------------------
def init_params():
    return {
        "w": jnp.array(random.random()),
        "b": jnp.array(0.0)
    }


# ----------------------------
# Model (simple linear model)
# y = wx + b
# ----------------------------
def model(params, x):
    return params["w"] * x + params["b"]


# ----------------------------
# Loss (mean squared error)
# ----------------------------
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)


# ----------------------------
# Gradient
# ----------------------------
grad_fn = jax.grad(loss_fn)


# ----------------------------
# Training loop
# ----------------------------
def train():
    params = init_params()

    # Fake dataset: y = 2x + 1
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([3.0, 5.0, 7.0, 9.0])

    lr = 0.1

    for i in range(50):
        grads = grad_fn(params, x, y)

        params["w"] -= lr * grads["w"]
        params["b"] -= lr * grads["b"]

        loss = loss_fn(params, x, y)

        print(f"Step {i}: loss={loss:.4f}, w={params['w']:.4f}, b={params['b']:.4f}")

    print("\nFinal params:", params)


if __name__ == "__main__":
    train()