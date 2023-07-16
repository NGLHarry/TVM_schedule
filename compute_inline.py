import tvm

n = 1024
k = 3
pad = 2
A = tvm.te.placeholder((n, n), name = "A")
W = tvm.te.placeholder((k, k), name = "W")
m = (n - k + 2 *pad) + 1

Apad = tvm.te.compute((n + 2 * pad, n + 2 * pad),
                      lambda yy, xx : tvm.te.if_then_else(
                          tvm.te.all(yy >= pad, yy < pad + n, xx >= pad, xx < pad+n),
                          A[yy-pad, xx - pad], tvm.te.const(0, "float32")),
                          name="Apad")
ry = tvm.te.reduce_axis((0, k), name = "ry")
rx = tvm.te.reduce_axis((0, k), name = "rx")

B = tvm.te.compute((m, m),
                   lambda yy, xx : tvm.te.sum(Apad[yy + ry, xx + rx] * W[ry, rx],
                                              axis=[ry, rx]),
                                              name="B")
s = tvm.te.create_schedule(B.op)

print(tvm.lower(s, [A, W, B], simple_mode=True))
print("---------cutting line---------")

s[Apad].compute_inline()

print(tvm.lower(s, [A, W, B], simple_mode=True))
exit(0)