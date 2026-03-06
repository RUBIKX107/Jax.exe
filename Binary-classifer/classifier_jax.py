import jax
import jax.numpy as jnp


# dataset
X = jnp.array([
    [1.,1.],
    [2.,1.],
    [5.,5.],
    [6.,5.]
])

y = jnp.array([0.,0.,1.,1.])


# initialize weights
key = jax.random.PRNGKey(0)

W = jax.random.normal(key,(2,))
b = jax.random.normal(key)


# sigmoid activation
def sigmoid(x):
    return 1/(1+jnp.exp(-x))


# model
def model(x,W,b):
    return sigmoid(jnp.dot(x,W) + b)


# loss function
def loss_fn(W,b,X,y):

    preds = model(X,W,b)

    return jnp.mean((preds - y)**2)


# gradients
grad_fn = jax.grad(loss_fn,argnums=(0,1))


lr = 0.1


for epoch in range(200):

    dW, db = grad_fn(W,b,X,y)

    W -= lr * dW
    b -= lr * db

    if epoch % 20 == 0:
        print("loss:",loss_fn(W,b,X,y))


print("\nPredictions:")
print(model(X,W,b))