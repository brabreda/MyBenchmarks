include("MyMapReduction.jl")

using CUDA
using GPUArrays
using BenchmarkTools


function benchmark_array()
    b = CUDA.rand(10000)

    v1 = @benchmark GPUArrays.mapreducedim!( x->x, +, similar(b,(1)),similar(b))
    #v2 = @benchmark mymapreducedim( x->x, +, similar(b,(1)), similar(b))

    #return v1, v2
end

function benchmark_2Dmatrix()
    for i in range(0,10)
        
    end
end

v1, v2 = benchmark_array()
# do something with v1, v2 like showing in graphs, etc.





