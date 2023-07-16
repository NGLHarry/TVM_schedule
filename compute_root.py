import tvm
n = 1024
A = tvm.te.placeholder((n,), name='A')
k = tvm.te.reduce_axis((0, n), 'k')
B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')

s = tvm.te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
BF = s.rfactor(B, ki)

tx = tvm.te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[BF].compute_root()

print(tvm.lower(s, [A, B], simple_mode=True))
exit(0)