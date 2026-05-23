import jax.numpy as jnp
import jax

def norm(X, eps=1e-8):
  X = X - X.mean(0)
  return X / (X.std(0) + eps)

arr = jnp.array([[[1,2,3],[1,1,1],[34,76,8]],[[1,2,-2],[1,6,1],[34,1,8]]])
arr1 = arr.reshape(6, 3) - arr.reshape(6,3).mean(0)
arr2 = arr1 / arr.reshape(arr1.shape).std(0)

