import tvm

n = 1024
m = 1024
A = tvm.te.placeholder((n, m), name="A")
k = tvm.te.reduce_axis((0, n), name="k")
l = tvm.te.reduce_axis((0, m), name="l")

B = tvm.te.compute((n,), lambda i : tvm.te.sum(A[i,l], axis=l), name="B")

s = tvm.te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor = 4)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].pragma(ki, "unroll")

print(tvm.lower(s, [A, B], simple_mode=True))