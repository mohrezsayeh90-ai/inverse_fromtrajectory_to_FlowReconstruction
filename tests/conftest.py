import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
jax.config.update('jax_enable_x64',True)
