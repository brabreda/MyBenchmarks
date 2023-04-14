using KernelAbstractions
using CUDA

@kernel function test(l)
        
        KernelAbstractions.@test(l)
end

# generic mapreduce
function mapreducedim(a)

    conf = KernelAbstractions.Config{
        10,
        length(a),
        10,
        10,
        10
    }

    if has_cuda_gpu()
        test(CUDABackend(), length(a))(conf, ndrange=length(a))
    end

    return 
end

a = CUDA.rand(10)
mapreducedim(a)

