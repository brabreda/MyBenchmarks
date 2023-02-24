include("MyMapReduction.jl")

using CUDA
using GPUArrays
using BenchmarkTools


function benchmark_array()
    a = CUDA.rand(10000)

    v1 = @benchmark GPUArrays.mapreducedim!( x->x, +, similar(a,(1)), similar(a))
    v2 = @benchmark mymapreducedim(similar(a,(1)), similar(a))

    return v1, v2
end

function benchmark_2Dmatrix()
    for i in range(0,10)
        
    end
end

v1, v2 = benchmark_array()
# do something with v1, v2 like showing in graphs, etc.
