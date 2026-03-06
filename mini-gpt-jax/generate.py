import jax.numpy as jnp
from model import MiniGPT

text = "hello world hello ai hello jax"

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

model = MiniGPT(vocab_size)

def generate(start_char, length=30):

    idx = char_to_ix[start_char]
    result = start_char

    for _ in range(length):

        probs = model.forward(idx)
        idx = int(jnp.argmax(probs))

        result += ix_to_char[idx]

    return result


print(generate("h"))