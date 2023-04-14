using CUDA
using GPUArrays
using BenchmarkTools
using DataFrames



function benchmark_baseline_1D()

    b =128 
    while b < 5_000_000
        println("benchmark: ",b)
        a = CUDA.rand(b)

        CUDA.@sync mapreduce(x->x, +, a);

        b = b * 2
    end 

    
    return 
end

function benchmark_KA_1D()
    a = CUDA.rand(10000);
    mapreducedim(x->x, +, similar(a,(1)), a; init=0.0);
    reductions = []

    grain = 2
    while grain <= 32
        b = 128
        println("\n", "##########################")
        println("grain: ",grain)
        println("##########################")
        while b < 5_000_000
            println("benchmark: ",b)
            a = CUDA.rand(b)
            c = similar(a,(1))

            mapreducedim(x->x, +, c, a; init=0.0, grain=grain);

            push!(reductions, c)
            

            b = b * 2
        end 
        grain = grain * 2
    end

    # println("\n","##########################")
    # println("result: ")
    # println("##########################")

    # c = 128
    # for i in reductions
    #     if c > 5_000_000
    #         print("\n")
    #         c = 128
    #     end

    #     print(i, "\t")

    #     c = c * 2
    # end
    
end

function benchmark_2Dmatrix()
    for i in range(0,10)
        
    end
end

#v1, v2 = benchmark_array()
#println(v1)
#println(v2)
# do something with v1, v2 like showing in graphs, etc.
