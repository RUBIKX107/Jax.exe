import jax
import jax.numpy as jnp

class MiniGPT:

    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):

        key = jax.random.PRNGKey(0)

        self.E = jax.random.normal(key,(vocab_size,embedding_dim))*0.1
        self.W1 = jax.random.normal(key,(embedding_dim,hidden_dim))*0.1
        self.W2 = jax.random.normal(key,(hidden_dim,vocab_size))*0.1

    def forward(self, x):

        embed = self.E[x]
        h = jnp.tanh(embed @ self.W1)
        logits = h @ self.W2

        return jax.nn.softmax(logits)