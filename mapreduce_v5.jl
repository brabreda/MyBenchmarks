using KernelAbstractions

struct Config{
    GROUPSIZE, 
    WARPSIZE,
    MAX_NDRANGE,

    ITEMS_PER_WORKITEM,
    USE_ATOMICS,
    USE_WARPS
    }

    function Config(groupsize, warpsize, max_ndrange, items_per_workitem , use_atomics, use_warps)
        new{groupsize, warpsize, max_ndrange, items_per_workitem, use_atomics, use_warps}()
    end
end

@inline function Base.getproperty(conf::Config{GROUPSIZE, WARPSIZE, MAX_NDRANGE, ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS}, sym::Symbol) where { GROUPSIZE, WARPSIZE, MAX_NDRANGE, ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS }
    if sym == :groupsize
        GROUPSIZE
    elseif sym == :warpsize
        WARPSIZE
    elseif sym == :max_ndrange
        MAX_NDRANGE
    elseif sym == :items_per_workitem
        ITEMS_PER_WORKITEM
    elseif sym == :use_atomics
        USE_ATOMICS
    elseif sym == :use_warps
        USE_WARPS        
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end

@enum GroupReduceAlgorithm  begin 
    THREADS
    WARP_WARP
    SEQUENTIAL_WARP
end

function get_reduce_config(::CUDABackend, op::OP, type::Type{T}) where {T, OP}
    dev = CUDA.device()
    #major = CUDA.capability(dev).major
    sm_count = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
    # if major >= ...
        # for SM ... groupreduce with threads is faster than any other algorithm
        return Config(512, 32, 512 * sm_count * 4, 32, false, true)
    # elseif major >= ...
    #     ...
    # else
    #     ...
    # end
end

# function get_reduce_config(::CUDABackend, conf)
#     dev = CUDA.device()
#     #major = CUDA.capability(dev).major
#     sm_count = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
#     # if major >= ...
#         # for SM ... groupreduce with threads is faster than any other algorithm
#         return Config(conf.groupsize, 32, 1024 * sm_count * 4, conf.items_per_workitem, false, false)
#     # elseif major >= ...
#     #     ...
#     # else
#     #     ...
#     # end
# end


@inline _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
@inline _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
@inline _map_getindex(args::Tuple{}, I) = ()

# we will still need some global temperary memory reduce between blocks
# NOTE: when every thread only needs to handle 1  the reduction wil just 
# result in heirachical block reductions

macro groupreduce(op, val, neutral, conf) 
    quote
        # use heuristic to determine the best algorithm only use groupreduce with only warps
        # if we groupsize is warpsize * warpsize. This way we don't have to use neutral elements.
        # This also seams to be faster for CUDA
        # if else will be optimized away by the compiler because the values are compile time constants
        algo = if $(esc(conf)).use_warps
            $(esc(conf)).warpsize * $(esc(conf)).warpsize == $(esc(conf)).groupsize ? WARP_WARP : SEQUENTIAL_WARP
        else 
            THREADS
        end
        # Using Val so that  algo is in the type domain and we can use it to #dispatch the right algorithm
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)), typeof($(esc(val))), Val(algo))
    end
end

@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{WARP_WARP::GroupReduceAlgorithm}) where {T}
    threadIdx_local = KernelAbstractions.@index(Local)
    groupsize = KernelAbstractions.@groupsize()[1]

    shared = KernelAbstractions.@localmem(T, 32)

    warpIdx, warpLane = fldmod1(threadIdx_local, 32)

    # each warp performs partial reduction
    val = KernelAbstractions.@warpreduce(op, val)

    # write reduced value to shared memory
    if warpLane == 1
        @inbounds shared[warpIdx] = val
    end

    # wait for all partial reductions
    KernelAbstractions.@synchronize()

    # read from shared memory only if that warp existed
    val = if threadIdx_local <= fld1(groupsize, 32)
            @inbounds shared[warpLane]
    else
        neutral
    end

    # final reduce within first warp
    if warpIdx == 1
        val =  KernelAbstractions.@warpreduce(op, val)
    end

    return val
end

@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{SEQUENTIAL_WARP::GroupReduceAlgorithm}) where {T}
    threadIdx_local = KernelAbstractions.@index(Local)
    groupsize = KernelAbstractions.@groupsize()[1]
    warpIdx, warpLane = fldmod1(threadIdx_local, 32)
    items_per_workitem = cld(groupsize, 32)

    shared = KernelAbstractions.@localmem(T, groupsize)

    @inbounds shared[threadIdx_local] = val 
    KernelAbstractions.@synchronize()

    if warpIdx == 1
        startIdx = ((threadIdx_local- 1) * items_per_workitem) + 1

        val = @inbounds shared[startIdx]
        index = startIdx + 1
        while index <= startIdx + items_per_workitem - 1
            val = op(val, shared[index])
            index += 1
        end

        val = KernelAbstractions.@warpreduce(op, val)
    end

    return val
end

@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{THREADS::GroupReduceAlgorithm}) where {T}
    threadIdx_local = KernelAbstractions.@index(Local)
    groupsize = KernelAbstractions.@groupsize()[1]
    
    shared = KernelAbstractions.@localmem(T, groupsize)

    @inbounds shared[threadIdx_local] = val

    # perform the reduction
    d = 1
    while d < groupsize
        KernelAbstractions.@synchronize()
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
    
    return val 
end



@kernel function reduce_kernel(f, op, neutral, R, A , conf)
    # values for the kernel
        threadIdx_local = @index(Local)
        threadIdx_global = @index(Global)
        groupIdx = @index(Group)
        gridsize = @ndrange()[1]

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
            val = op(val,f(A[index]))
            index += gridsize
        end

        # reduce every block to a single value
        val = @groupreduce(op, val, neutral, conf)

        # use helper function to deal with atomic/non atomic reductions
        if threadIdx_local == 1
            if conf.use_atomics
                #CUDA.@atomic R[1] = op(R[1], val)
            else
                @inbounds R[groupIdx] = val
            end
        end
end

@inline function __use_atomics(__ctx__, op, val, R, ::Val{false})
    threadIdx_local = @index(Local)
    groupIdx = @index(Group)

    if threadIdx_local == 1
        @inbounds R[groupIdx] = val
    end
end

@inline function __use_atomics(__ctx__, op, val, R, ::Val{true})
    threadIdx_local = @index(Local)

    if threadIdx_local == 1
        CUDA.@atomic R[1] = op(R[1], val)
    end
end

# generic mapreduce
function mapreducedim!(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted}; init=nothing, conf=nothing) where {F, OP}

    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end
    
    backend = KernelAbstractions.get_backend(A) 

    conf = if conf == nothing get_reduce_config(backend, op, eltype(A)) else conf end

    return __devicereduce(f, op, R, A, init, conf, backend, Val(length(R)))
end

@inline function __devicereduce(f, op, R, A::Union{AbstractArray,Broadcast.Broadcasted},init ,conf, backend, ::Val{1})
    if length(A) <= conf.items_per_workitem * conf.groupsize

        reduce_kernel(backend, conf.groupsize)(f, op, init, R, A, conf, ndrange=conf.groupsize)
        return R
    else
        # determine the number of workitems
        gridsize = cld(length(A), conf.items_per_workitem)
        gridsize = min(gridsize, conf.max_ndrange)

        groups = cld(gridsize, conf.groupsize)
        partial = conf.use_atomics==true ? R : similar(R, (size(R)...,groups))

        reduce_kernel(backend, conf.groupsize)(f, op, init, partial, A, conf, ndrange=gridsize)
        
        if !conf.use_atomics
            __devicereduce(x->x, op, R, partial,init,conf, backend,Val(1))
            return R
        end

        return R
    end
end


println(2^19)
a = CUDA.ones(2^19)
b= similar(a,(1))

#@show b
#b = CUDA.ones(1)
# # get type of a
# println(typeof(a))


#conf = get_reduce_config(CUDABackend(), +, Float32)
#println(mapreduce(x->x, +, a)) 
println(mapreducedim!(x->x, +, b,a; init=Float32(0.0)))
@benchmark CUDA.@sync( begin mapreducedim!(x->x, +, $b, $a; init=Float32(0.0)) end)



#println(mapreduce(x->x, +, a)) 
#println(mapreducedim(x->x, +, similar(a,(1)), a, b; init=Float32(0.0), warps=true))