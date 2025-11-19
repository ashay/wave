from sympy import Max, Min, ceiling

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)


# This function provides a GEMM kernel where the workgroups executing the kernel are re-arranged
# to provide potential L2 Cache Optimizations. More details can be found in docs/wave/workgroup_reordering.rst
def get_reordered_matmul(
    m_size: int,
    n_size: int,
    k_size: int,
    block_m_size: int,
    block_n_size: int,
    block_k_size: int,
    param_w: int,
    group_m_size: int,
    mfma_variant,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    PARAM_W = tkl.sym.PARAM_W
    # The grouping factor to group columns by in our reordering scheme
    GROUP_SIZE_M = tkl.sym.GROUP_SIZE_M
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M // 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N // 2)]

    # From Algorithm 1 in the paper titled HipKittens: Fast and Furious AMD Kernels
    workgroups_in_dim_0 = ceiling(M / BLOCK_M)
    workgroups_in_dim_1 = ceiling(N / BLOCK_N)
    blocks = workgroups_in_dim_0 * workgroups_in_dim_1
    xy = WORKGROUP_1 * workgroups_in_dim_0 + WORKGROUP_0
    num_xcds = 8
    blocks_per_cycle = num_xcds * GROUP_SIZE_M
    limit = (blocks // blocks_per_cycle) * blocks_per_cycle
    xy = Min(xy, limit)
    xcd = xy % num_xcds
    local = xy // num_xcds
    chunk_idx = local // GROUP_SIZE_M
    pos = local % GROUP_SIZE_M
    xy = chunk_idx * blocks_per_cycle + xcd * GROUP_SIZE_M + pos

    tid_per_group = PARAM_W * workgroups_in_dim_1
    group_id = xy // tid_per_group
    first_row = group_id * PARAM_W
    win_h = Min(workgroups_in_dim_0 - first_row, PARAM_W)
    l = xy % tid_per_group
    new_wg0 = first_row + (l % win_h)
    new_wg1 = l // win_h

    constraints += [tkw.ReorderingConstraint(new_wg0, 0)]
    constraints += [tkw.ReorderingConstraint(new_wg1, 1)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        M: m_size,
        N: n_size,
        K: k_size,
        BLOCK_M: block_m_size,
        BLOCK_N: block_n_size,
        BLOCK_K: block_k_size,
        PARAM_W: param_w,
        GROUP_SIZE_M: group_m_size,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
    }
    hyperparams.update(get_default_scheduling_params())
    return gemm, hyperparams
