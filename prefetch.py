import tvm

n = 1024
dtype = "float32"
k = tvm.te.reduce_axis((0, n), name="k")
A = tvm.te.placeholder((n, n), dtype=dtype, name="A")
B = tvm.te.compute((n,), lambda i: tvm.te.sum(A[i, k], axis=k), name="B")

s = tvm.te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].prefetch(A, s[B].op.reduce_axis[0], 1)
print(tvm.lower(s, [A, B], simple_mode=True))