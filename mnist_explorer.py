import jax.numpy as jnp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt 

# ── Load MNIST ──────────────────────────────────────────
ds = tfds.load("mnist", split="train", as_supervised=True)

# Convert to numpy arrays first, then JAX
images, labels = [], []
for img, lbl in ds.take(1000):   # grab 1000 samples
    images.append(img.numpy())
    labels.append(lbl.numpy())

# Stack into JAX arrays
X = jnp.array(images)            # shape: (1000, 28, 28, 1)
y = jnp.array(labels)            # shape: (1000,)

# ── Inspect ──────────────────────────────────────────────
print("Shape  :", X.shape)
print("Dtype  :", X.dtype)
print("Min    :", X.min())
print("Max    :", X.max())
print("Labels :", jnp.unique(y))
# ── Visualize 5x5 grid ───────────────────────────────────
fig, axes = plt.subplots(5, 5, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].squeeze(), cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_grid.png")    # saves to file since no display in Codespaces
print("Saved mnist_grid.png")
# ── Pixel distribution ───────────────────────────────────
flat_pixels = X.flatten()        # JAX flatten to 1D

plt.figure(figsize=(8, 4))
plt.hist(flat_pixels, bins=50, color="steelblue", edgecolor="none")
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel value (0-255)")
plt.ylabel("Count")
plt.savefig("pixel_dist.png")
print("Saved pixel_dist.png")