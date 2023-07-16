import tvm

n = 1024
A = tvm.te.placeholder((n,), name='A')
k = tvm.te.reduce_axis((0, n), name='k')

B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')

s = tvm.te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))