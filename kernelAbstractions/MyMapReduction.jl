using CUDA
using Cthulhu
using KernelAbstractions
using CUDAKernels

@kernel function small_reduce_kernel(R,A)
    gridIdx = @index(Global) # dit was local maar dit werkte niet bij voor een array die kleiner is dan 32 * 1024
    val = 0.0
    
    reduce_id = gridIdx
    while reduce_id <= length(A)
        val += A[reduce_id]
        reduce_id += (length(R) * @groupsize()[1])
    end
    
    val = @reduce(+ , val)

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

function mymapreducedim(f::F, op::OP, R::AnyCuArray{T},
                        A::Union{AbstractArray,Broadcast.Broadcasted}) where {T,F,OP}

    Base.check_reducedims(R, A)
    # Check whether R has compatible dimensions w.r.t. A for reduction, komt uit base.jl.

    length(A) == 0 && return R # isempty(::Broadcasted) iterates
    dev = device()

    # Use shuffle instructions if T is one of followed elements
    # be conservative about using shuffle instructions
    shuffle = T <: Union{Bool,
                            UInt8, UInt16, UInt32, UInt64, UInt128,
                            Int8, Int16, Int32, Int64, Int128,
                            Float16, Float32, Float64,
                            ComplexF16, ComplexF32, ComplexF64}

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R)) 
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

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

