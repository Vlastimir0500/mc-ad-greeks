import jax.numpy as jnp

def call(S, K):
    return jnp.maximum(S - K, 0.0)

def put(S, K):
    return jnp.maximum(K - S, 0.0)

def digital_call(S, K):
    return (S > K).astype(S.dtype)
