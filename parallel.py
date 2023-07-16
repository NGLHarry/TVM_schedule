import tvm

m = 1024
n = 1024

A = tvm.te.placeholder((n,m), name="A")
l = tvm.te.reduce_axis((0,m), name="l")

B = tvm.te.compute((n,), lambda i : tvm.te.sum(A[i,l], axis=l), name="B")

s = tvm.te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].parallel(B.op.reduce_axis[0])
print(tvm.lower(s, [A, B], simple_mode=True))