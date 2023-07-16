schedule与计算逻辑分离是自动代码生成技术的核心概念，由MIT CASIL组的[Jonathan Ragan-Kelley](http://people.csail.mit.edu/jrk/)在2012年发表在SIGGRAPH上的文章率先提出，然后在2013年发表在PLDI上的文章给出了schedule的精确定义：
>* 1.When and where should be the value at each coordinate in each function be computed?
>* 2.Where should they be stored?
>* 3.How long are values cached and communicated across multiple consumers, and when are they independently recomputed by each?

第一条是描述了数据计算顺序对性能的影响，第二条是数据的存储位置对性能影响，最后一条是多线程处理过程中，不同线程数据应该如何进行交互。

事实上，**schedule就是一系列优化选择的集合**。这些**选择不会影响计算的结果**，但是**由于其包含着对architecture的理解，因此对性能是至关重要的**。往往一个选择或者多个选择的组合，都会被称为schedule。
常用的Schedule有：
### 存储层次的相关Schedule
 * 1 **cache_read(tensor, scope, readers)**
将数据存储到片上缓存，减少访问数据时间。
>cache_read将tensor读入指定存储层次scope的cache，这个设计的意义在于显式利用现有计算设备的on-chip memory hierarchy。这个例子中（`AA = s.cache_read(A, "shared", [B])`），会先将A的数据load到shared memory中，然后计算B。在这里，我们需要引入一个stage的概念，一个op对应一个stage，也就是通过cache_read会新增一个stage。
```python
import tvm

n = 1024
dtype = "float32"
A = tvm.te.placeholder((n, n), dtype = dtype, name="A")
k = tvm.te.reduce_axis((0,n), name = "k")
B = tvm.te.compute((n,), lambda i: tvm.te.sum(A[i,k], axis = k), name="B")

s = tvm.te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
print("-------------")
AA = s.cache_read(A, "shared", [B])

print(tvm.lower(s, [A, B], simple_mode=True))
```
结果如下：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (k: int32, 0, 1024) {
      B[i] = (B[i] + A[((i*1024) + k)])
    }
  }
}


-------------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  allocate(A.shared: Pointer(shared float32), float32, [1048576]), storage_scope = shared {
    for (ax0: int32, 0, 1024) {
      for (ax1: int32, 0, 1024) {
        let cse_var_1: int32 = ((ax0*1024) + ax1)
        A.shared_1: Buffer(A.shared, float32, [1048576], [], scope="shared")[cse_var_1] = A[cse_var_1]
      }
    }
    for (i: int32, 0, 1024) {
      B[i] = 0f32
      for (k: int32, 0, 1024) {
        B[i] = (B[i] + A.shared_1[((i*1024) + k)])
      }
    }
  }
}
```
* 2 **cache_write(tensor, scope)**
将结果写入片上缓存，然后再写入片外缓存。当然这里的片上和片外并不是绝对的概念，也可以理解为不同层次的存储结构。
> cache_write和cache_read对应，是先在shared memory中存放计算结果，最后将结果写回到global memory。当然在真实的场景中，我们往往是会将结果先放着register中，最后写回。
```python
import tvm

n = 1024
dtype = "float32"
A = tvm.te.placeholder((n, n), dtype = dtype, name="A")
k = tvm.te.reduce_axis((0,n), name = "k")
B = tvm.te.compute((n,), lambda i: tvm.te.sum(A[i,k], axis = k), name="B")

s = tvm.te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("-------------")

AA = s.cache_write(B, "local")

print(tvm.lower(s, [A, B], simple_mode=True))
```
执行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (k: int32, 0, 1024) {
      B[i] = (B[i] + A[((i*1024) + k)])
    }
  }
}


-------------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  allocate(B.local: Pointer(local float32), float32, [1024]), storage_scope = local {
    for (i.c: int32, 0, 1024) {
      B.local_1: Buffer(B.local, float32, [1024], [], scope="local")[i.c] = 0f32
      for (k: int32, 0, 1024) {
        B.local_1[i.c] = (B.local_1[i.c] + A[((i.c*1024) + k)])
      }
    }
    for (i: int32, 0, 1024) {
      B[i] = B.local_1[i]
    }
  }
}
```


* 3 **set_scope**
为数据指定存储位置，相比于cache_read和cache_write提供了更灵活的指定数据存储方式。本质上是相同的。
>set_scope指定stage计算结果所在的存储层次，为tensor选择最优的存储位置，适用于设置线程间的共享内存。事实上，set_scope是cache_read的子操作。

```python
import tvm

n = 1024
dtype = "float32"
A = tvm.te.placeholder((n, n), dtype=dtype, name='A')
k = tvm.te.reduce_axis((0, n), name='k')
B = tvm.te.compute((n,), lambda i: tvm.te.sum(A[i, k], axis=k), name='B')
C = tvm.te.compute((n,), lambda i: B[i] + 10, name='C')

s = tvm.te.create_schedule(C.op)

print(tvm.lower(s, [A, C], simple_mode=True))
print("---------cutting line---------")

s[B].set_scope('shared')

print(tvm.lower(s, [A, C], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024], [])} {
  allocate(B: Pointer(global float32), float32, [1024]), storage_scope = global {
    for (i: int32, 0, 1024) {
      B_1: Buffer(B, float32, [1024], [])[i] = 0f32
      for (k: int32, 0, 1024) {
        B_1[i] = (B_1[i] + A[((i*1024) + k)])
      }
    }
    for (i_1: int32, 0, 1024) {
      C[i_1] = (B_1[i_1] + 10f32)
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024], [])} {
  allocate(B: Pointer(shared float32), float32, [1024]), storage_scope = shared {
    for (i: int32, 0, 1024) {
      B_1: Buffer(B, float32, [1024], [], scope="shared")[i] = 0f32
      for (k: int32, 0, 1024) {
        B_1[i] = (B_1[i] + A[((i*1024) + k)])
      }
    }
    for (i_1: int32, 0, 1024) {
      C[i_1] = (B_1[i_1] + 10f32)
    }
  }
}
```

* 4 **storage_align**
在我看的文章中，storage_align是针对GPU shared memory的一个优化，目的是为了减少同一个bank的访问冲突。在GPU中shared memory被分割成多个bank，这些bank可以被独立线程同时访问。Storage_align就是为了将数据和bank大小匹配，减少bank conflict的发生。AI芯片中也有类似的问题，只有尽量减少bank冲突的发生，才能最大化并行计算。
> storage_align把stage对应的存储空间以factor为单位、以offset为偏置重新对齐，以避免GPU共享访问时的bank conflict，关于bank conflict可以参考[2](https://devblogs.nvidia.com/using-shared-memory-cuda-cc/)。

```python
import tvm

n = 1024
factor = 100
offset = 8
dtype = "float32"
A = tvm.te.placeholder((n, n), dtype=dtype, name="A")
k = tvm.te.reduce_axis((0, n), name="k")
B = tvm.te.compute((n,), lambda i : tvm.te.sum(A[i, k], axis=k),name="B")

s = tvm.te.create_schedule(B.op)

AA = s.cache_read(A, "shared",[B])
print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[AA].storage_align(AA.op.axis[0], factor, offset)
print(tvm.lower(s, [A, B], simple_mode=True)) 
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  allocate(A.shared: Pointer(shared float32), float32, [1048576]), storage_scope = shared {
    for (ax0: int32, 0, 1024) {
      for (ax1: int32, 0, 1024) {
        let cse_var_1: int32 = ((ax0*1024) + ax1)
        A.shared_1: Buffer(A.shared, float32, [1048576], [], scope="shared")[cse_var_1] = A[cse_var_1]
      }
    }
    for (i: int32, 0, 1024) {
      B[i] = 0f32
      for (k: int32, 0, 1024) {
        B[i] = (B[i] + A.shared_1[((i*1024) + k)])
      }
    }
  }
}


---------cutting line---------
compute(A.shared, body=[A[ax0, ax1]], axis=[iter_var(ax0, range(min=0, ext=1024)), iter_var(ax1, range(min=0, ext=1024))], reduce_axis=[], tag=, attrs={})
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  allocate(A.shared: Pointer(shared float32), float32, [1134592]), storage_scope = shared {
    for (ax0: int32, 0, 1024) {
      for (ax1: int32, 0, 1024) {
        A.shared_1: Buffer(A.shared, float32, [1134592], [], scope="shared")[((ax0*1108) + ax1)] = A[((ax0*1024) + ax1)]
      }
    }
    for (i: int32, 0, 1024) {
      B[i] = 0f32
      for (k: int32, 0, 1024) {
        B[i] = (B[i] + A.shared_1[((i*1108) + k)])
      }
    }
  }
}
```

* 5 **compute_at**
不懂CUDA，所以对文章中的代码不是很理解，但是从其解释看，对于多次循环的计算（或者多维计算），可以通过并行计算来降维。
>compute_at将当前的stage附着到目标stage的指定iter方向上，同时与目标stage采用相同的并行方式，在其内部完成当前stage的计算。往往compute_at会与cache_read和cache_write一起使用。
```python
import tvm

n = 1024
A = tvm.te.placeholder((n,), name="A")
k = tvm.te.reduce_axis((0,n), name="k")
B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name="B")

s = tvm.te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor = 32)
BF = s.rfactor(B, ki)

tx = tvm.te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))

```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  allocate(B.rf: Pointer(global float32), float32, [32]), storage_scope = global {
    for (k.inner: int32, 0, 32) {
      B.rf_1: Buffer(B.rf, float32, [32], [])[k.inner] = 0f32
      for (k.outer: int32, 0, 32) {
        B.rf_1[k.inner] = (B.rf_1[k.inner] + A[((k.outer*32) + k.inner)])
      }
    }
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local {
      attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
      @tir.tvm_thread_allreduce(1u32, B.rf_1[threadIdx.x], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
      B[0] = reduce_temp0_1[0]
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
  allocate(B.rf: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local {
    B.rf_1: Buffer(B.rf, float32, [1], [], scope="local", align=4)[0] = 0f32
    for (k.outer: int32, 0, 32) {
      B.rf_1[0] = (B.rf_1[0] + A[((k.outer*32) + threadIdx.x)])
    }
    attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
    @tir.tvm_thread_allreduce(1u32, B.rf_1[0], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
    B[0] = reduce_temp0_1[0]
  }
}
```

* 6 **compute_inline**
将独立操作转化为内联函数，有点类似FPGA上的流水线计算。转化成内联函数从上层层面减少了stage。在FPGA中也有类似问题，可以将具有相同迭代的多条指令放在一起执行。
> compute_inline把独立的计算操作转化成内联函数形式，在使用到原计算结果时再调用内联函数完成运算，通过compute_inline来减少一个stage。
```python
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
```
运行结果：
```
@main = primfn(A_1: handle, W_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             W: Buffer(W_2: Pointer(float32), float32, [9], []),
             B: Buffer(B_2: Pointer(float32), float32, [1052676], [])}
  buffer_map = {A_1: A, W_1: W, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), W_1: W_3: Buffer(W_2, float32, [3, 3], []), B_1: B_3: Buffer(B_2, float32, [1026, 1026], [])} {
  allocate(Apad: Pointer(global float32), float32, [1056784]), storage_scope = global {
    for (yy: int32, 0, 1028) {
      for (xx: int32, 0, 1028) {
        Apad_1: Buffer(Apad, float32, [1056784], [])[((yy*1028) + xx)] = @tir.if_then_else(((((2 <= yy) && (yy < 1026)) && (2 <= xx)) && (xx < 1026)), A[(((yy*1024) + xx) - 2050)], 0f32, dtype=float32)
      }
    }
    for (yy_1: int32, 0, 1026) {
      for (xx_1: int32, 0, 1026) {
        B[((yy_1*1026) + xx_1)] = 0f32
        for (ry: int32, 0, 3) {
          for (rx: int32, 0, 3) {
            let cse_var_1: int32 = ((yy_1*1026) + xx_1)
            B[cse_var_1] = (B[cse_var_1] + (Apad_1[((((yy_1*1028) + (ry*1028)) + xx_1) + rx)]*W[((ry*3) + rx)]))
          }
        }
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, W_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             W: Buffer(W_2: Pointer(float32), float32, [9], []),
             B: Buffer(B_2: Pointer(float32), float32, [1052676], [])}
  buffer_map = {A_1: A, W_1: W, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), W_1: W_3: Buffer(W_2, float32, [3, 3], []), B_1: B_3: Buffer(B_2, float32, [1026, 1026], [])} {
  for (yy: int32, 0, 1026) {
    for (xx: int32, 0, 1026) {
      B[((yy*1026) + xx)] = 0f32
      for (ry: int32, 0, 3) {
        for (rx: int32, 0, 3) {
          let cse_var_3: int32 = (yy + ry)
          let cse_var_2: int32 = (xx + rx)
          let cse_var_1: int32 = ((yy*1026) + xx)
          B[cse_var_1] = (B[cse_var_1] + (@tir.if_then_else(((((2 <= cse_var_3) && (cse_var_3 < 1026)) && (2 <= cse_var_2)) && (cse_var_2 < 1026)), A[(((((yy*1024) + (ry*1024)) + xx) + rx) - 2050)], 0f32, dtype=float32)*W[((ry*3) + rx)]))
        }
      }
    }
  }
}
```
* 7 **compute_root**
Compute_at的反操作。
> compute_root是compute_at的反操作。因为不做任何schedule的话，每一个stage默认就是compute_root的，这个schedule相当于注释了对之前对一个stage的compute操作。
```python
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
```

运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
  allocate(B.rf: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local {
    B.rf_1: Buffer(B.rf, float32, [1], [], scope="local", align=4)[0] = 0f32
    for (k.outer: int32, 0, 32) {
      B.rf_1[0] = (B.rf_1[0] + A[((k.outer*32) + threadIdx.x)])
    }
    attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
    @tir.tvm_thread_allreduce(1u32, B.rf_1[0], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
    B[0] = reduce_temp0_1[0]
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  allocate(B.rf: Pointer(global float32), float32, [32]), storage_scope = global {
    for (k.inner: int32, 0, 32) {
      B.rf_1: Buffer(B.rf, float32, [32], [])[k.inner] = 0f32
      for (k.outer: int32, 0, 32) {
        B.rf_1[k.inner] = (B.rf_1[k.inner] + A[((k.outer*32) + k.inner)])
      }
    }
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local {
      attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
      @tir.tvm_thread_allreduce(1u32, B.rf_1[threadIdx.x], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
      B[0] = reduce_temp0_1[0]
    }
  }
}
```

### 常见循环优化
* 8 **fuse**
将多个循环iter融合为一个iter。
> fuse用于融合两个iter，将两层循环合并到一层，其返回值为iter类型，可以多次合并。

```python
import tvm

n = 1024
A = tvm.te.placeholder((n,), name="A")
k = tvm.te.reduce_axis((0, n), name = "k")

B = tvm.te.compute((1,), lambda i : tvm.te.sum(A[k], axis=k), name="B")

s = tvm.te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor = 32)
print(tvm.lower(s,[A, B], simple_mode=True))
print("---------cutting line---------")

s[B].fuse(ko, ki)

print(tvm.lower(s, [A, B], simple_mode=True))
```
运行结果
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  B[0] = 0f32
  for (k.outer: int32, 0, 32) {
    for (k.inner: int32, 0, 32) {
      B[0] = (B[0] + A[((k.outer*32) + k.inner)])
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  B[0] = 0f32
  for (k.outer.k.inner.fused: int32, 0, 1024) {
    B[0] = (B[0] + A[k.outer.k.inner.fused])
  }
}
```

* 9 **split**
Fuse的反操作，将一次循环迭代拆分为多次。
> split是fuse的反操作，把iter以factor为间隔分离成outer与inner两层迭代，增加循环层数，用于将循环操作分割为更小的子任务。事实上，以CUDA为例，gridDim和blockDim都可以最多是三维，所以通过split可以产生新的维度用于绑定到grid和block上[3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)。

```python
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
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  B[0] = 0f32
  for (k: int32, 0, 1024) {
    B[0] = (B[0] + A[k])
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  B[0] = 0f32
  for (k.outer: int32, 0, 32) {
    for (k.inner: int32, 0, 32) {
      B[0] = (B[0] + A[((k.outer*32) + k.inner)])
    }
  }
}
```

* 10 **reorder**
调整循环计算迭代顺序。
> reorder用于重置循环iter的内外顺序，根据局部性原理，最大化利用cache中的现有数据，减少反复载入载出的情况。注意，这里到底怎样的顺序是最优化的是一个很有趣的问题。以矩阵乘法为例，M, N, K三维，往往是将K放在最外层可以最大程度利用局部性。

```python
import tvm

n = 1024
A = tvm.te.placeholder((n, n), name="A")
B = tvm.te.placeholder((n, n), name="B")
C = tvm.te.compute((n, n), lambda i,j : A[i, j] + B[i, j], name="C")

s = tvm.te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor = 32)
yo, yi = s[C].split(s[C].op.axis[1], factor = 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].reorder(xo, yo, yi, xi)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (i.outer: int32, 0, 32) {
    for (i.inner: int32, 0, 32) {
      for (j.outer: int32, 0, 32) {
        for (j.inner: int32, 0, 32) {
          let cse_var_1: int32 = ((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)
          C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])
        }
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (i.outer: int32, 0, 32) {
    for (j.outer: int32, 0, 32) {
      for (j.inner: int32, 0, 32) {
        for (i.inner: int32, 0, 32) {
          let cse_var_1: int32 = ((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)
          C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])
        }
      }
    }
  }
}
```

* 11 **tile**
Tile也是将循环迭代进行拆分，拆分多次计算。是split+reorder。
> tile将stage的两个维度按照各自的factor拆分，并以固定顺序依次返回两个outer和两个inner的iter，从而增加循环层数，形成更小的计算任务。事实上，tile是可以由split和reorder来实现的，tile是矩阵乘法和卷积计算的重要schedule。

```python
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
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (i: int32, 0, 1024) {
    for (j: int32, 0, 1024) {
      C[((i*1024) + j)] = 0f32
      for (K: int32, 0, 1024) {
        let cse_var_2: int32 = (i*1024)
        let cse_var_1: int32 = (cse_var_2 + j)
        C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + K)]*B[((K*1024) + j)]))
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (i.outer: int32, 0, 32) {
    for (j.outer: int32, 0, 32) {
      for (i.inner: int32, 0, 32) {
        for (j.inner: int32, 0, 32) {
          C[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = 0f32
          for (K: int32, 0, 1024) {
            let cse_var_3: int32 = (j.outer*32)
            let cse_var_2: int32 = ((i.outer*32768) + (i.inner*1024))
            let cse_var_1: int32 = ((cse_var_2 + cse_var_3) + j.inner)
            C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + K)]*B[(((K*1024) + cse_var_3) + j.inner)]))
          }
        }
      }
    }
  }
}
```

* 12 **unroll**
将循环展开，增加并发执行。
> unroll是一种常见的循环优化方法，减分支预测失败减少，如果循环体内语句没有数据相关，增加了并发执行的机会，也有利于指令流水线的调度[4](https://en.wikipedia.org/wiki/Loop_unrolling)。

```python
import tvm

n = 1024
A = tvm.te.placeholder((n, n), name="A")
B = tvm.te.placeholder((n, n), name="B")
C = tvm.te.compute((n, n), lambda i, j : A[i, j] + B[i,j], name="C")

s = tvm.te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor = 4)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].unroll(xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (i.outer: int32, 0, 256) {
    for (i.inner: int32, 0, 4) {
      for (j: int32, 0, 1024) {
        let cse_var_1: int32 = (((i.outer*4096) + (i.inner*1024)) + j)
        C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (i.outer: int32, 0, 256) {
    for (j: int32, 0, 1024) {
      let cse_var_1: int32 = ((i.outer*4096) + j)
      C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])
    }
    for (j_1: int32, 0, 1024) {
      let cse_var_2: int32 = (((i.outer*4096) + j_1) + 1024)
      C[cse_var_2] = (A[cse_var_2] + B[cse_var_2])
    }
    for (j_2: int32, 0, 1024) {
      let cse_var_3: int32 = (((i.outer*4096) + j_2) + 2048)
      C[cse_var_3] = (A[cse_var_3] + B[cse_var_3])
    }
    for (j_3: int32, 0, 1024) {
      let cse_var_4: int32 = (((i.outer*4096) + j_3) + 3072)
      C[cse_var_4] = (A[cse_var_4] + B[cse_var_4])
    }
  }
}
```

### 多线程并行优化
* 13 **vectorize**
将循环迭代替换成ramp，可以通过SIMD指令实现数据批量计算，也就是单指令多数据计算。这在AI加速中会很常用，每条指令都是多数据计算的。
> vectorize把iter方向上的循环迭代替换成ramp，从而通过SIMD指令实现数据的批量计算，并且只有在数据size为常数、且分割的iter为2的幂（即满足SIMD的计算数量）时才会发生替换，否则vectorize没有效果，是SIMD计算设备的常用schedule。
```python
import tvm

M = 1024
N = 1024
A = tvm.te.placeholder((M, N), name='A')
B = tvm.te.placeholder((M, N), name='B')
C = tvm.te.compute(
           (M, N),
           lambda x, y: A[x, y] + B[x, y],
           name='C')

s = tvm.te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].vectorize(yi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```
运行结果
```
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner: int32, 0, 32) {
        for (y.inner: int32, 0, 32) {
          let cse_var_1: int32 = ((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)
          C[cse_var_1] = (A[cse_var_1] + B[cse_var_1])
        }
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner: int32, 0, 32) {
        let cse_var_1: int32 = (((x.outer*32768) + (x.inner*1024)) + (y.outer*32))
        C[ramp(cse_var_1, 1, 32)] = (A[ramp(cse_var_1, 1, 32)] + B[ramp(cse_var_1, 1, 32)])
      }
    }
  }
}
```

* 14 **bind**
CUDA中使用的优化方法，将iter绑定到不同线程，实现并发计算。
> bind将iter绑定到block或thread的index上，从而把循环的任务分配到线程，实现并行化计算，这是针对CUDA后端最核心的部分。
```python
import tvm

n = 1024
A = tvm.te.placeholder((n,), name="A")
k = tvm.te.reduce_axis((0, n), name="k")

B = tvm.te.compute((1, ), lambda i : tvm.te.sum(A[i], axis=k), name="B")

s = tvm.te.create_schedule(B.op)

ko, ki = s[B].split(s[B].op.reduce_axis[0], factor = 32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].bind(ko, tvm.te.thread_axis("blockIdx.x"))
s[B].bind(ki, tvm.te.thread_axis("threadIdx.x"))

print(tvm.lower(s, [A, B], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  B[0] = 0f32
  for (k.outer: int32, 0, 32) {
    for (k.inner: int32, 0, 32) {
      B[0] = (B[0] + A[0])
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 32;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
    attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
    @tir.tvm_thread_allreduce(1u32, A[0], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], blockIdx.x, threadIdx.x, dtype=handle)
    B[0] = reduce_temp0_1[0]
  }
}
```

* 15 **parallel**
实现多设备并行.
> parallel将指定iter的for循环替换为parallel操作，从而在GPU以外的CPU等设备上实现并行。
```python
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
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (l: int32, 0, 1024) {
      B[i] = (B[i] + A[((i*1024) + l)])
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (l: int32, 0, 1024) "parallel" {
      B[i] = (B[i] + A[((i*1024) + l)])
    }
  }
}
```

### 其他schedule
* 16 **pragma**
可以在代码中人为添加编译注释，人为干预编译优化。HLS中就是通过这样的方式来实现c的硬件编程的。
> pragma用于添加编译注释，使编译器遵循pragma的要求，实现unroll, vectorize等调度功能。事实上一个新的优化规则，都可以看做是一种gragma，也被称作directive[5](https://en.wikipedia.org/wiki/Directive_(programming))。

```python
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
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (l.outer: int32, 0, 256) {
      for (l.inner: int32, 0, 4) {
        B[i] = (B[i] + A[(((i*1024) + (l.outer*4)) + l.inner)])
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (l.outer: int32, 0, 256) {
      let cse_var_1: int32 = ((i*1024) + (l.outer*4))
       {
        B[i] = (B[i] + A[cse_var_1])
        B[i] = (B[i] + A[(cse_var_1 + 1)])
        B[i] = (B[i] + A[(cse_var_1 + 2)])
        B[i] = (B[i] + A[(cse_var_1 + 3)])
      }
    }
  }
}
```

* 17 **prefetch**
将数据计算和load后者store数据重叠起来，在FPGA中是很常见优化方法。
> prefetch利用数据的空间局部性，用于使得前一个iter的计算与后一个iter的访存overlap起来，以提高访存和计算的并行度，减少耗时。本质上是软件流水线的概念，不是硬件prefetch。
```python
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
```
运行结果
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (k: int32, 0, 1024) {
      B[i] = (B[i] + A[((i*1024) + k)])
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  for (i: int32, 0, 1024) {
    B[i] = 0f32
    for (k: int32, 0, 1024) {
      for (prefetch.A.1: int32, 0, 1) {
        for (prefetch.A.0: int32, 0, 1) {
          @tir.prefetch(@tir.address_of(A[(((k*1024) + i) + 1024)], dtype=handle), 0, 3, 1, dtype=float32)
        }
      }
      B[i] = (B[i] + A[((i*1024) + k)])
    }
  }
}
```


* 18 **tensorize**
将tensor作为一个整体匹配硬件的计算核心，比如一个卷积运算就可以实现在FPGA上的一个匹配。
>tensorize将计算作为整体，编译为一个tensor_intrin函数中。这是因为很多计算属于常用计算，针对这些计算已经有了很好的built-in的schedule，通过tensorize可以直接调用这些内置的intrinsic，其实这也就是intrinsic在计算机科学中的本意[6](https://en.wikipedia.org/wiki/Intrinsic_function)。

```python
import tvm
from tvm import te

N, M, L = 1024, 512, 64
A = tvm.te.placeholder((N, L), name='A')
B = tvm.te.placeholder((M, L), name='B')
k = tvm.te.reduce_axis((0, L), name='k')
C = tvm.te.compute((N, M), lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k), name='C')
s = tvm.te.create_schedule(C.op)


def intrin_gemv(m, l):
    a = te.placeholder((l,), name="a")
    b = te.placeholder((m, l), name="b")
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((m,), lambda i: te.sum(a[k] * b[i, k], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "gemv_update",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                bb.access_ptr("r"),
                m,
                l,
                bb.strides[0],
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

factor = 16
x, y = C.op.axis
z, = C.op.reduce_axis
yo, yi = s[C].split(y, factor=factor)
s[C].reorder(x, yo, yi, z)

gemv = intrin_gemv(factor, L)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].tensorize(yi, gemv)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [65536], []),
             B: Buffer(B_2: Pointer(float32), float32, [32768], []),
             C: Buffer(C_2: Pointer(float32), float32, [524288], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 64], []), B_1: B_3: Buffer(B_2, float32, [512, 64], []), C_1: C_3: Buffer(C_2, float32, [1024, 512], [])} {
  for (i: int32, 0, 1024) {
    for (j.outer: int32, 0, 32) {
      for (j.inner: int32, 0, 16) {
        C[(((i*512) + (j.outer*16)) + j.inner)] = 0f32
        for (k: int32, 0, 64) {
          let cse_var_1: int32 = (((i*512) + (j.outer*16)) + j.inner)
          C[cse_var_1] = (C[cse_var_1] + (A[((i*64) + k)]*B[(((j.outer*1024) + (j.inner*64)) + k)]))
        }
      }
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [65536], []),
             B: Buffer(B_2: Pointer(float32), float32, [32768], []),
             C: Buffer(C_2: Pointer(float32), float32, [524288], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 64], []), B_1: B_3: Buffer(B_2, float32, [512, 64], []), C_1: C_3: Buffer(C_2, float32, [1024, 512], [])} {
  for (i: int32, 0, 1024) {
    for (j.outer: int32, 0, 32) {
      @tir.call_extern("gemv_update", @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C_2, ((i*512) + (j.outer*16)), 16, 2, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), A_2, (i*64), 64, 1, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), B_2, (j.outer*1024), 1024, 1, dtype=handle), 16, 64, 64, dtype=int32)
    }
  }
}
```
* 19 **rfactor(tensor, axis, factor_axis=0)**
> rfactor对原tensor在axis方向以factor_axis为间隔做reduction操作。
```python
import tvm

n = 1024
k = tvm.te.reduce_axis((0, n), name="k")

A = tvm.te.placeholder((n, ), name = "A")
B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis = k), name="B")

s = tvm.te.create_schedule(B.op)
ko, ki = s[B].split(s[B].op.reduce_axis[0], 32)

print(tvm.lower(s, [A,B], simple_mode=True))
print("---------cutting line---------")

BR = s.rfactor(B, ki)

print(tvm.lower(s, [A, B], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  B[0] = 0f32
  for (k.outer: int32, 0, 32) {
    for (k.inner: int32, 0, 32) {
      B[0] = (B[0] + A[((k.outer*32) + k.inner)])
    }
  }
}


---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  allocate(B.rf: Pointer(global float32), float32, [32]), storage_scope = global {
    for (k.inner: int32, 0, 32) {
      B.rf_1: Buffer(B.rf, float32, [32], [])[k.inner] = 0f32
      for (k.outer: int32, 0, 32) {
        B.rf_1[k.inner] = (B.rf_1[k.inner] + A[((k.outer*32) + k.inner)])
      }
    }
    B[0] = 0f32
    for (k.inner.v: int32, 0, 32) {
      B[0] = (B[0] + B.rf_1[k.inner.v])
    }
  }
}
```

* 20 **set_store_predicate**
> set_store_predicate设置了store的条件，适用于在多线程调度中预防写操作之间的冲突。
```python
import tvm

n = 1024
A = tvm.te.placeholder((n,), name='A')
k = tvm.te.reduce_axis((0, n), 'k')
B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')

s = tvm.te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
tx = tvm.te.thread_axis("threadIdx.x")
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].set_store_predicate(tx.var.equal(0))

print(tvm.lower(s, [A, B], simple_mode=True))
```
运行结果：
```
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  allocate(B.rf: Pointer(global float32), float32, [1]), storage_scope = global {
    B[0] = 0f32
    for (k.inner.v: int32, 0, 16) {
      B.rf_1: Buffer(B.rf, float32, [1], [], align=4)[0] = 0f32
      for (k.outer: int32, 0, 64) {
        B.rf_1[0] = (B.rf_1[0] + A[((k.outer*16) + k.inner.v)])
      }
      B[0] = (B[0] + B.rf_1[0])
    }
  }
}
 git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

---------cutting line---------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024], []), B_1: B_3: Buffer(B_2, float32, [1], [])} {
  allocate(B.rf: Pointer(global float32), float32, [1]), storage_scope = global {
    B[0] = 0f32
    for (k.inner.v: int32, 0, 16) {
      B.rf_1: Buffer(B.rf, float32, [1], [], align=4)[0] = 0f32
      for (k.outer: int32, 0, 64) {
        B.rf_1[0] = (B.rf_1[0] + A[((k.outer*16) + k.inner.v)])
      }
      if (threadIdx.x: int32 == 0) {
        B[0] = (B[0] + B.rf_1[0])
      }
    }
  }
}
```


* 21 **create_group(outputs, inputs, include_inputs=False)**
> create_group对从inputs到outputs的所有stage创建group，group本质上是一个虚拟stage，可以通过操作这个虚拟stage来一起操作这个group里的所有stage。本例中，通过compute_at使这个group中的D和E，一起附着到指定操作中。
```python
import tvm

n = 1024
k = tvm.te.reduce_axis((0, n), name='k')

A = tvm.te.placeholder((n, n), name='A')
B = tvm.te.placeholder((n, n), name='B')

D = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='D')
E = tvm.te.compute((n, n), lambda i, j: D[i, j] + B[i, j], name='E')
F = tvm.te.compute((n,), lambda i: tvm.te.sum(E[i, k], axis=k), name='F')

s = tvm.te.create_schedule(F.op)

print(tvm.lower(s, [A, B, E], simple_mode=True))
print("---------cutting line---------")

g = s.create_group(outputs = E, inputs = [A, B], include_inputs=True)
g.compute_at(s[F], F.op.reduce_axis[0])

print(tvm.lower(s, [A, B, E], simple_mode=True))
```
运行结果：
```
import tvm

n = 1024
k = tvm.te.reduce_axis((0, n), name='k')

A = tvm.te.placeholder((n, n), name='A')
B = tvm.te.placeholder((n, n), name='B')

D = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='D')
E = tvm.te.compute((n, n), lambda i, j: D[i, j] + B[i, j], name='E')
F = tvm.te.compute((n,), lambda i: tvm.te.sum(E[i, k], axis=k), name='F')

s = tvm.te.create_schedule(F.op)

print(tvm.lower(s, [A, B, E], simple_mode=True))
print("---------cutting line---------")

g = s.create_group(outputs = E, inputs = [A, B], include_inputs=True)
g.compute_at(s[F], F.op.reduce_axis[0])

print(tvm.lower(s, [A, B, E], simple_mode=True))
```

相关示例代码参考https://github.com/StrongSpoon/tvm.schedule


参考：
[tvm schedule详细举例](https://zhuanlan.zhihu.com/p/94846767)