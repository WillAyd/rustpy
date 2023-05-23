This code corresponds to the blog post at https://willayd.com/comparing-cython-to-rust-evaluating-python-extensions.html

To get the cython file working you can do roughly the following. You will need to change the include / link / python version paths to match your system

```sh
cythonize -3 cypy.pyx
gcc -g3 -Wall -Werror -fPIC -shared -I/home/willayd/mambaforge/envs/scratchpad/include/python3.11 -DNPY_NO_DEPRECATED_API=0 -I/home/willayd/mambaforge/envs/scratchpad/lib/python3.11/site-packages/numpy/core/include cypy.c -o cypy.so -L/home/willayd/mambaforge/envs/scratchpad/lib/ -lpython3.11
```

For the rust extension a simple ``maturin develop --release`` will do

Some sample code for benchmarking:

```python
import numpy as np
np.random.seed(42)
arr = np.random.randint(100_000, size=(100, 1_000_000))
```

```python
import cypy
import rustpy

result1 = cypy.find_max(arr)
result2 = rustpy.find_max(arr)
result3 = rustpy.find_max_parallel(arr)
result4 = rustpy.find_max_unsafe(arr)

((result1 == result2) & (result1 == result3) & (result1 == result4)).all()
```
