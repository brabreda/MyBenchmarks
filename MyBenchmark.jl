using CUDA
using GPUArrays
using BenchmarkTools
using DataFrames


function benchmark_baseline()
    w = []
    v = []

    b =128 
    while b < 10_000_000
        a = CUDA.rand(b)
        push!(w, b)

        time = CUDA.@elapsed GPUArrays.mapreducedim!(x->x, +, similar(a,(1)), a)

        push!(v, time)

        b = b * 2
    end

    
    return DataFrame(N=w, Time=v)
end

function benchmark_array()
    a = CUDA.rand(10000)

    v1 = GPUArrays.mapreducedim!( x->x, +, similar(a,(1)), a)
    v2 = mymapreducedim( x->x, +, similar(a,(1)), a)

    return v1, v2
end

function benchmark_2Dmatrix()
    for i in range(0,10)
        
    end
end

#v1, v2 = benchmark_array()
#println(v1)
#println(v2)
# do something with v1, v2 like showing in graphs, etc.
