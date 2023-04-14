using KernelAbstractions
using CUDA



@inline _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
@inline _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
@inline _map_getindex(args::Tuple{}, I) = ()

# we will still need some global temperary memory reduce between blocks
# NOTE: when every thread only needs to handle 1  the reduction wil just 
# result in heirachical block reductions
@kernel function reduce_kernel(f, op, neutral, conf, R, A)
    # values for the kernel
    threadIdx_local = @index(Local)
    groupIdx = @index(Group)
    threads_per_group = @groupsize()[1]

    startIdx_block = groupIdx * conf.elements_per_thread * conf.groupsize
    grain = 2

    # if @index(Global) == 1
    #     @print("threadIdx_local: ", threadIdx_local, "\n","groupIdx: ", groupIdx, "\n", "elements_per_thread: ", elements_per_thread, "\n", "elements_per_group: ", elements_per_group, "\n", length(A),"\n")

    # end

    # load neutral value
    neutral = if neutral === nothing
        R[@index(Global)]
    else
        neutral
    end
    
    val = op(neutral, neutral)
    # every thread reduces a few values parrallel
    d = 0
    while  d < conf.elements_per_thread
        index = (@index(Global)-1) * conf.elements_per_thread + d + 1
        if index <= length(A)
            val = op(val,A[index])
        end
        d += 1
    end

    # reduce every block to a single value
    @synchronize()
    val = @reduce(op, val, neutral, conf)

    # write reduces value to memory
    if threadIdx_local == 1
        R[groupIdx] = val
    end
end

@kernel function reduce_kernel(f, op, neutral, conf, Rreduce, Rother, R, A...)
    # when we reduce to a lower dim we have to take into account that only one block 
    # can acces a slice 
    threadIdx_reduce = @index(Local)
    blockDim_reduce = @groupsize()[1]
    blockIdx_reduce, blockIdx_other = fldmod1(@index(Group), length(Rother))
    gridDim_reduce = fld(@ndrange()[1],blockDim_reduce) ÷ length(Rother)


    #@print("blockIdx_reduce: ", blockIdx_reduce)
    #@print("blockIdx_other: ", blockIdx_other)

    # hold the lane of the reduction. this is espacelly important when length(Rother) != 1
    # it needs to make sure that threads within a lane dont add values of another lane
    iother = blockIdx_other
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., blockIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end
        
        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a block
        ireduce = threadIdx_reduce + (blockIdx_reduce - 1) * blockDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(_map_getindex(A, J)...)) 
            ireduce += blockDim_reduce * gridDim_reduce
        end

        val = @reduce(op, val, neutral, conf)

        # write back to memory
        if threadIdx_reduce == 1
            R[Iout] = val
        end
    end
end

# generic mapreduce
function mapreducedim(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted}; init=nothing , grain=2) where {F, OP}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # get the backend based on the array
    backend = KernelAbstractions.get_backend(A)

    # TODO: don't hard code these values
    threads = 1024
    maxthreads = 250 * 1024
    
    # when reducing to a single value, we can use a specialized kernel that works without cartesianindices
    # TODO: condition should be changed because it can also depend on f and op op should be associative
    if length(R) == 1

        groupsize = length(A) > 1024 ? 1024 : length(A)
        # when the length of array is larger than maxthreads * grain, every thread should reduce more values

        # when all the values fit in 1 block the elements per block should be 1 this will result in hierachical
        # block reductions 
        elements_per_thread = if length(A) > maxthreads * grain 
            grain
        else 
            cld(length(A), groupsize) != 1 ? grain : 1
        end 
        groups_per_ndrange = cld(length(A) , elements_per_thread * groupsize)

        # @show(elements_per_thread)
        # @show(groups_per_ndrange)

        # reduction config
        conf = KernelAbstractions.Config{
            32,

            groupsize,
            groups_per_ndrange,
            groups_per_ndrange * groupsize,

            elements_per_thread,
            
            KernelAbstractions.supports_atomics(backend),
            KernelAbstractions.supports_lanes(backend)
        }

        # only a single block needs to be launched
        if conf.groups_per_ndrange == 1
            reduce_kernel(backend, conf.groupsize)(f, op, init, conf, R, A, ndrange=conf.ndrange)
            return R
        else
            partial = similar(R, (size(R)...,  conf.groups_per_ndrange))
            reduce_kernel(backend, conf.groupsize)(f, op, init, conf, partial, A, ndrange=conf.ndrange)

            mapreducedim(f, op, R, partial; init=init)
            return R
        end
    
    else
        # we have to use  CartesianIndices

        # iteration domain, split in two: one part covers the dimensions that should
        # be reduced, and the other covers the rest. combining both covers all values.
        Rall = CartesianIndices(axes(A))
        Rother = CartesianIndices(axes(R))
        Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

        # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
        #       CartesianIndices object with UnitRanges that behave badly on the GPU.
        @assert length(Rall) == length(Rother) * length(Rreduce)
        @assert length(Rother) > 0

        blocks = if length(R) == 1 
            cld(length(A),threads)
        else
            length(Rother)
        end

        conf = KernelAbstractions.Config{
            32,
            threads,
            length(R) == 1 ? threads * cld(length(A),threads) : threads * length(Rother),
            KernelAbstractions.supports_atomics(CUDABackend()),
            KernelAbstractions.supports_lanes(CUDABackend())
        }

        # allocate an additional, empty dimension to write the reduced value to.
        if has_cuda_gpu()
            if length(R) == 1 
                partial = similar(R, (size(R)..., fld(conf.ndrange , conf.threads_per_block)))
                reduce_kernel(backend, conf.threads_per_block)(f, op, init, conf, Rreduce, Rother, partial, A, ndrange=conf.ndrange)

                print(partial)
            else
                reduce_kernel(backend, conf.threads_per_block)(f, op, init, conf, Rreduce, Rother, R, A, ndrange=conf.ndrange)

                print(R[1])
            end 
            
        end
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    # Rall = CartesianIndices(axes(A))
    # Rother = CartesianIndices(axes(R))
    # Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

    # # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    # #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    # @assert length(Rall) == length(Rother) * length(Rreduce)
    # @assert length(Rother) > 0

    
    return R
end

a = CUDA.rand(4_000_000)

mapreducedim(x->x, +, similar(a,(1)), a; init=0.0, grain=8);

 

