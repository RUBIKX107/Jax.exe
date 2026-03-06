import jax
import jax.numpy as jnp


# training data
x = jnp.array([1.,2.,3.,4.,5.])
y_true = 2*x + 1


# initialize parameters
key = jax.random.PRNGKey(0)

w = jax.random.normal(key)
b = jax.random.normal(key)


# model
def model(x,w,b):
    return w*x + b


# loss function
def loss_fn(w,b,x,y):

    pred = model(x,w,b)

    return jnp.mean((pred - y)**2)


# gradient function
grad_fn = jax.grad(loss_fn,argnums=(0,1))


learning_rate = 0.01


for epoch in range(200):

    dw, db = grad_fn(w,b,x,y_true)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 20 == 0:
        loss = loss_fn(w,b,x,y_true)
        print("epoch:",epoch,"loss:",loss)


print("\nLearned parameters:")
print("w =",w)
print("b =",b)