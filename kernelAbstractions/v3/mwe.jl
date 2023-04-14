using KernelAbstractions
using CUDA



@inline _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
@inline _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
@inline _map_getindex(args::Tuple{}, I) = ()




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

    # write back to memory
    if @index(Global) < length(R)
        R[thea] = val
        #@synchronize()
        # if @index(Global) == 1
        #      @print(val,"\n")
        # end
    end
  


    grain = 2
    # perform a reduction
    d = 1
    while d < length(R)
        @synchronize()
        index = grain * d * (@index(Global)-1) + 1
        @inbounds if index <= length(R)
            
            val = neutral
            other_val = if index + d <= length(R)
                R[index+d]
            else
                neutral
            end


            val = R[index]   

            R[index] = op(R[index], other_val)

            if d == 64
                @print("indexen:", index, " - " ,index + d , "\n","verschil:" , other_val, " + ", val, " = ", R[index], "\n", "\n")
            end 

            if d == 128
                @print("indexen:", index, " - " ,index + d , "\n","verschil:" , other_val, " + ", val, " = ", R[index], "\n", "\n")
            end 
        end
        d *= grain
    end

    if @index(Global) == 1
        @print( R[1], "\n")
    end
end

# generic mapreduce
function mapreducedim(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted}; init=nothing) where {F, OP}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

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

    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)
    @assert length(Rother) > 0

    threads = 1024

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
            reduce_kernel(CUDABackend(), conf.threads_per_block)(f, op, init, conf, Rreduce, Rother, partial, A, ndrange=conf.ndrange)

            #
            print(partial)
        else
            reduce_kernel(CUDABackend(), conf.threads_per_block)(f, op, init, conf, Rreduce, Rother, R, A, ndrange=conf.ndrange)

            print(R[1])
        end 
        
    end

    return R
end

using CUDA

a = CUDA.rand(250000)

println(mapreduce(x->x, +,a))

println(mapreducedim(x->x, +, similar(a,(1)), a; init=0.0))

 

