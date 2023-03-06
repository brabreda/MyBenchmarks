using CUDA

@inline function reduce_block(val::T) where T
    # shared mem for partial sums
    shared = CuStaticSharedArray(T, 1)

    wid, lane = fldmod1(threadIdx().x, warpsize())
    # each warp performs partial reduction
    val =  CUDA.reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        CUDA.atomic_add!(shared[1], val)
    end

    if lane == 1
    end

    # final reduce within first warp
    if wid == 1
        val = shared[1]
    end
    
    return val
end

# er wordt maar 1 block gelaunched
function small_reduce_kernel(R,A)
    gridIdx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    val = 0.0
    
    reduce_id = gridIdx
    while reduce_id <= length(A)
        val += A[reduce_id]
        reduce_id += (length(R) * blockDim().x)
    end
    
    val = reduce_block(val)

    return
end

function big_reduce_kernel(R,A)
    gridIdx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

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

    
    if(length(A) > threads * groups )
        @cuda threads=threads blocks=groups big_reduce_kernel(partial, A)
        @cuda threads=threads blocks=1 small_reduce_kernel(R,partial)
    else 
        @cuda threads=threads blocks=groups small_reduce_kernel(R,A)       
    end     
    
    return R
end

