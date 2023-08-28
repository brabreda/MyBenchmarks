

@inline function Base.getproperty(conf::Type{Config{ WARPSIZE, MAX_GROUPSIZE, MAX_CONCURRENCY, SUPPORTS_ATOMICS, SUPPORTS_WARP_REDUCE }}, sym::Symbol) where { WARPSIZE, MAX_GROUPSIZE, MAX_CONCURRENCY, SUPPORTS_ATOMICS, SUPPORTS_WARP_REDUCE }
    if sym == :warpsize
        WARPSIZE
    elseif sym == :max_groupsize
        MAX_GROUPSIZE
    elseif sym == :max_concurrency
        MAX_CONCURRENCY
    elseif sym == :supports_atomics
        SUPPORTS_ATOMICS
    elseif sym == :supports_warp_reduce
        SUPPORTS_WARP_REDUCE        
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end

@inline _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
@inline _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
@inline _map_getindex(args::Tuple{}, I) = ()

@inline function __reduce( op, val, neutral, ::Type{T}, supports_warp_reduce) where {T}

    groupsize = CUDA.blockDim().x
    threadIdx_local = CUDA.threadIdx().x

    # if supports_warp_reduce
    #     shared = CUDA.CuStaticSharedArray(T, 32)

    #     warpIdx, warpLane = fldmod1(threadIdx_local, 32)

    #     # each warp performs partial reduction
    #     val = CUDA.reduce_warp(op, val)

    #     # write reduced value to shared memory
    #     if warpLane == 1
    #         @inbounds shared[warpIdx] = val
    #     end

    #     # wait for all partial reductions
    #     CUDA.sync_threads()

    #     # read from shared memory only if that warp existed
    #     val = if threadIdx_local <= fld1(groupsize, 32)
    #             @inbounds shared[warpLane]
    #     else
    #         neutral
    #     end

    #     # final reduce within first warp
    #     if warpIdx == 1
    #         val =  CUDA.reduce_warp(op, val)
    #     end
    # else
        shared = CUDA.CuStaticSharedArray(T, 1024)

        @inbounds shared[threadIdx_local] = val

        # perform the reduction
        d = 1
        while d < groupsize
            CUDA.sync_threads()
            index = 2 * d * (threadIdx_local-1) + 1
            @inbounds if index <= groupsize
                other_val = if index + d <= groupsize
                    shared[index+d]
                else
                    neutral
                end
                shared[index] = op(shared[index], other_val)
            end
            d *= 2
        end

        # load the final value on the first thread
        if threadIdx_local == 1
            val = @inbounds shared[threadIdx_local]
        end
    # end
    
    return val 
end

# we will still need some global temperary memory reduce between blocks
# NOTE: when every thread only needs to handle 1  the reduction wil just 
# result in heirachical block reductions
function reduce_kernel(f, op, neutral, grain, R, A, supports_warp_reduce)
    # values for the kernel
    threadIdx_local = CUDA.threadIdx().x
    threadIdx_global = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    groupIdx = CUDA.blockIdx().x
    gridsize = CUDA.gridDim().x * CUDA.blockDim().x

    # load neutral value
    neutral = if neutral === nothing
        R[1]
    else
        neutral
    end
    
    val = op(neutral, neutral)

    # every thread reduces a few values parrallel
    index = threadIdx_global 
    while index <= length(A)
        val = op(val,A[index])
        index += gridsize
    end

    # reduce every block to a single value
    val = __reduce(op, val, neutral, Float32, supports_warp_reduce)

    # write reduces value to memory
    if threadIdx_local == 1
        R[groupIdx] = val
    end

    return nothing
end

# generic mapreduce
function mapreducedim(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted}, partial; init=nothing, atomics = false, warps = false, conf=nothing) where {F, OP}

    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end
    
    dev = CUDA.device()
    
    major = CUDA.capability(dev).major
    max_groupsize = if major >= 3 1024 else 512 end
    gridsize = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    max_concurrency = max_groupsize * gridsize
    # supports_atomics = if warps == nothing major >= 2 else atomics end
    # supports_warp_reduce = if warps == nothing major >= 5 else warps end
    supports_atomics = atomics
    supports_warp_reduce = warps
    
    conf = Config{
        32,
        max_groupsize,
        max_concurrency,
        supports_atomics,
        supports_warp_reduce
    }

    if length(R) == 1
        # only a single block needs to be launched
        if length(A) <= max_groupsize
            
            @cuda threads=max_groupsize blocks=1 reduce_kernel(f, op, init, 1, R, A, warps)
        
            return R
        else
            
            groups_per_grid = cld(length(A), max_groupsize)
            groups_per_grid = min(groups_per_grid, gridsize)

            #partial = similar(R, (size(R)..., supports_atomics ? 1 : groups_per_grid))
            @cuda threads=max_groupsize blocks=groups_per_grid reduce_kernel( f, op, init, 1, partial, A, warps)
            
            if !supports_atomics 
                mapreducedim(f, op, R, partial,partial; atomics=atomics,warps=warps, init=init, conf=conf)
                return R
            end

            return partial
        end
    
    else
        # # we have to use  CartesianIndices

        # # iteration domain, split in two: one part covers the dimensions that should
        # # be reduced, and the other covers the rest. combining both covers all values.
        # Rall = CartesianIndices(axes(A))
        # Rother = CartesianIndices(axes(R))
        # Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

        # # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
        # #       CartesianIndices object with UnitRanges that behave badly on the GPU.
        # @assert length(Rall) == length(Rother) * length(Rreduce)
        # @assert length(Rother) > 0

        # blocks = if length(R) == 1 
        #     cld(length(A),threads)
        # else
        #     length(Rother)
        # end

        # # allocate an additional, empty dimension to write the reduced value to.
        # if length(R) == 1 
        #     partial = similar(R, (size(R)..., fld(conf.ndrange , conf.threads_per_block)))
        #     reduce_kernel(backend, group)(f, op, init, conf, Rreduce, Rother, partial, A, ndrange=conf.ndrange)

        #     print(partial)
        # else
        #     reduce_kernel(backend, conf.threads_per_block)(f, op, init, conf, Rreduce, Rother, R, A, ndrange=conf.ndrange)

        #     print(R[1])
        # end 
    end

    return R
end

#  a = CUDA.ones(2^16)
#  b = CUDA.ones(64)
# # # get type of a
# # println(typeof(a))

# println(mapreduce(x->x, +, a)) 
# println( mapreducedim(x->x, +, similar(a,(1)),a,b; init=Float32(0.0)))


# a = CUDA.ones(60_000)

# println(mapreduce(x->x, +, a)) 
# println(mapreducedim(x->x, +, similar(a,(1)), a; init=Float32(0.0), warps))