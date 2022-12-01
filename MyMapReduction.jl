using GPUArrays
using CUDA
import CUDA: @device_override
using KernelAbstractions
using CUDAKernels



GPUArrays.mapreducedim!(f::F, op::OP, R::AnyCuArray{T}, A::Union{AbstractArray,Broadcast.Broadcasted}; 
    init=nothing) where {F, OP, T} = 
    invoke(mymapreducedim, Tuple{AnyCuArray{T}, Union{AbstractArray,Broadcast.Broadcasted}}, R, A)

function reduce_SIMD(val)
    offset = 0x00000001
    while offset < @SIMDsize()

        val += @SIMDreduce(0xffffffff, val, offset)
        offset <<= 1
    end

    return val
end

function reduce_group(val::T) where T
    shared = @localmem T (32,1)

    Sid, lane = fldmod1(@index(Local), @SIMDsize())

    val = reduce_SIMD(val)

    # Enkel de eerste lane van iedere SIMD unit mag de waarde naar shared memory schrijven
    if lane == 1
        @inbounds shared[Sid] = val
    end

    @synchronize()

    # de eerste 32 values worden in val gestoken, als er er minder dan 32 warps passen in 1 block
    # dan vullen we de rest op met nullen
    val = if @index(Local) <= fld1(@groupsize()[1], @SIMDsize())
            @inbounds shared[lane]
    else
        0
    end

    # final reduce within first warp
    if Sid == 1
        val = reduce_SIMD(val)
    end
    return val

end

@kernel function small_reduce_kernel(R,A)
    shared = @localmem Float64 (32,1)
    gridIdx = @index(Local)
    val = 0

    reduce_id = gridIdx
    while reduce_id <= length(A)
        val += A[reduce_id]
        reduce_id += (length(R) * @groupsize()[1])
    end
    
    val = reduce_group(val)

    if gridIdx == 1
        R[1] = val
    end 
end

@kernel function big_reduce_kernel(R,A)

    gridIdx = @index(Global)

    val = 0

    reduce_id = gridIdx
    while reduce_id <= length(A)
        val += A[reduce_id]
        reduce_id += (length(R) * @groupsize()[1])
    end

    R[gridIdx] = val
end

function mymapreducedim( R::AnyCuArray{T},
                        A::Union{AbstractArray,Broadcast.Broadcasted}) where {T}

    # Check whether R has compatible dimensions w.r.t. A for reduction
    Base.check_reducedims(R, A)

    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    threads = 1024
    groups = 32
    partial = similar(R, threads * groups)

    if has_cuda_gpu()
        if(length(R) > threads * groups )
            big_reduce_kernel(CUDADevice(), threads)(partial, A, ndrange=threads*groups)
        end

        small_reduce_kernel(CUDADevice(), threads)(R,partial, ndrange=threads)
    end
    print(R[1])
end

