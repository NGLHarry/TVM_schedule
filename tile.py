import tvm

n = 1024
A = tvm.te.placeholder((n, n), name="A")
B = tvm.te.placeholder((n, n), name="B")
K = tvm.te.reduce_axis((0, n), name="K")
C = tvm.te.compute((n, n), lambda i,j : tvm.te.sum(A[i,K] * B[K, j], axis=K), name="C")

s = tvm.te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))