using GPUArrays
using CUDA
using KernelAbstractions
using CUDAKernels
using Cthulhu


#function change_map_reduce()
#    GPUArrays.mapreducedim!(f::F, op::OP, R::AnyCuArray{T}, A::Union{AbstractArray,Broadcast.Broadcasted}; 
#        init=nothing) where {F, OP, T} = 
#        invoke(mymapreducedim, Tuple{AnyCuArray{T}, Union{AbstractArray,Broadcast.Broadcasted}}, R, A)
#end

@kernel function small_reduce_kernel(R,A)
    gridIdx = @index(Global) # dit was local maar dit werkte niet bij voor een array die kleiner is dan 32 * 1024
    val = 0.0

    reduce_id = gridIdx
    while reduce_id <= length(A)
        val += A[reduce_id]
        reduce_id += (length(R) * @groupsize()[1])
    end
   
    val = @reduce(val)

    if gridIdx == 1
        R[1] = val
    end 
end

@kernel function big_reduce_kernel(R,A)

    gridIdx = @index(Global)

    val = 0.0

    reduce_id = gridIdx
    while reduce_id <= length(A)
        val += A[reduce_id]
        reduce_id += length(R)
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
        if(length(A) > threads * groups )
            event = big_reduce_kernel(CUDADevice(), threads)(partial, A, ndrange=threads*groups)
            event = small_reduce_kernel(CUDADevice(), threads)(R,partial, ndrange=threads, dependencies=(event,))
            wait(event)
        else 
            event = small_reduce_kernel(CUDADevice(), threads)(R,A, ndrange=threads)
            wait(event)
        end
    end
    return R
end
