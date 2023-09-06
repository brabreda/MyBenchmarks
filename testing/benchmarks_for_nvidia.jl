using GPUArrays
using DataFrames, Tables, Statistics, CSV

using BenchmarkTools

# include benchmark_to_df.jl
include("benchmarks_to_df.jl")

function benchmark_CUDA(warps)
    results = []
    N = []
    
    n =128 
    while n < 5_000_000 
        data=CUDA.ones(n)
        final=CUDA.ones(1)

        dev = CUDA.device()
    
        # this will take longer as every iteration the function will be parsed
        bench = @benchmarkable CUDA.@sync( begin GPUArrays.mapreducedim!(x->x, +, $final, $data; init=Float32(0.00),shuffle=$warps) end) evals=1 samples=10000 seconds = 10000

        result = run(bench)
        display(result)
 
        # add result to results
        push!(results, result)
        push!(N, n)

        n = n * 2

        sleep(30)

    end 

    df_benchmark = mapreduce(vcat, zip(results, N)) do (x, y)
        df = DataFrame(x)
        df.N .= y
        df
    end

    #return df_benchmark
    return df_benchmark
end

function benchmark_KA(__warps, __atomics,; __groupsizes = [1024], __items_per_workitem = [1], __groups_multiplier = [1], op = +)
    groupsize = __groupsizes
    items_per_workitem = __items_per_workitem
    groups_multiplier = __groups_multiplier

    for g in groupsize
        @show g
        for i in items_per_workitem
            @show i
            for m in groups_multiplier
                @show m
                n = 128

                while n < 5_000_000 
                    data = CUDA.ones(n)
                    final = CUDA.ones(1)

                    dev = CUDA.device()
        
                    major = CUDA.capability(dev).major
                    max_groupsize = if major >= 3 1024 else 512 end
                    gridsize = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                    max_concurrency = max_groupsize * gridsize
                    # supports_atomics = if warps == nothing major >= 2 else atomics end
                    # supports_warp_reduce = if warps == nothing major >= 5 else warps end
                    supports_atomics = atomics
                    supports_warp_reduce = warps

                    max_ndrange = g * gridsize * m

                    conf = Config(g,32, max_ndrange, i, supports_atomics, supports_warp_reduce)
                    #@show conf

                    results = []
                    N = []
                    groupsizes = []
                    items_per_workitems = []
                    groups_multipliers = []


                    # this will take longer as every iteration the function will be parsed
            
                    for idk in 1:3
                        bench = @benchmarkable CUDA.@sync( begin mapreducedim!(x->x, op, $final, $data; init=Float32(0.00),conf=$conf) end) evals=1 samples=500 seconds = 10000

                        result = run(bench)

                        display(result)
            
                        # add result to results
                        push!(results, result)
                        push!(N, n)
                        push!(groupsizes, g)
                        push!(items_per_workitems, i)
                        push!(groups_multipliers, m)
                    end
            
                    n = n * 2
            
                    data = nothing
                    partial = nothing
                    
                    CUDA.reclaim()

                    df_benchmark = mapreduce(vcat, zip(results, N, groupsizes,items_per_workitems, groups_multipliers)) do (x,y, groupsizes,items_per_workitems, groups_multipliers )
                        df = DataFrame(x)
                        df.N .= y
                        df.groupsize .= groupsizes
                        df.items_per_workitem .= items_per_workitems
                        df.groups_multiplier .= groups_multipliers
                        df
                    end
    
                    df_benchmark.atomics .= atomics
                    df_benchmark.warps .= warps
                    df_benchmark.package .= "KernelAbstractions.jl"
        
                    CSV.write("KA.csv", df_benchmark, append=true)                  
                end         
            end
        end
    end 
end

function benchmark_error(;neutral=Float32(0.0),type=Float32, groupsize=1024, items_per_workitem=1, groups_multiplier=1, op=+, warps=false, atomics=false)
    n = 128
    while n < 5_000_000 
        data = CUDA.rand(n)
        final = CUDA.rand(1)

        dev = CUDA.device()

        major = CUDA.capability(dev).major
        max_groupsize = if major >= 3 1024 else 512 end
        gridsize = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        max_concurrency = max_groupsize * gridsize
        supports_atomics = atomics
        supports_warp_reduce = warps
        max_ndrange = groupsize * gridsize * groups_multiplier

        conf = Config(groupsize,32, max_ndrange, items_per_workitem, supports_atomics, supports_warp_reduce)
        x = 0
        # create array with size 10000
        array = zeros(10000)
        for i in 1:10000
            data = CUDA.rand(type,n)
            final = CUDA.rand(type,1)

            CUDA.@sync( begin totalka = mapreducedim!(x->x, op, final, data; init=neutral,conf=conf) end)
            CUDA.@allowscalar totalka = totalka[1]

            totalcuda = mapreduce(x->x, op, data)
            array[i] = abs(totalka - totalcuda)  
        end
        println("mean for $n: \t", mean(array))
        n= n*2
    end
   
end


            
