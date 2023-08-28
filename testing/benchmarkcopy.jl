using GPUArrays
using NVTX
using DataFrames, Tables, Statistics, CSV
using CUDA

include("../mapreduce.jl")

using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000

# TODO add delta between the reduction results
# with tables we can implement a custom table type for BenchmarkTools.Trial
Tables.istable(::Type{<:BenchmarkTools.Trial}) = true
Tables.columnaccess(::Type{<:BenchmarkTools.Trial}) = true
Tables.columns(m::BenchmarkTools.Trial) = m
Tables.columnnames(m::BenchmarkTools.Trial) = [:times, :gctimes, :memory, :allocs]
Tables.schema(m::BenchmarkTools.Trial) = Tables.Schema(Tables.columnnames(m), (Float64, Float64, Int, Int))
function Tables.getcolumn(m::BenchmarkTools.Trial, i::Int)
    i == 1 && return m.times
    i == 2 && return m.gctimes
    i == 3 && return fill(m.memory, length(m.times))
    return fill(m.allocs, length(m.times))
end
Tables.getcolumn(m::BenchmarkTools.Trial, nm::Symbol) = Tables.getcolumn(m, nm == :times ? 1 : nm == :gctimes ? 2 : nm == :memory ? 3 : 4)

# add optinal dims to use 1 function for all reductions
# mapreduce(x->x, +, data)
function benchmark_KA(atomics, warps)

    results = []
    N = []
    
    n =128 
    while n < 5_000_000 

        # this will take longer as every iteration the function will be parsed
        result = @benchmark CUDA.@sync( mapreducedim(x->x, *, similar(data,(1)), data; init=Float32(0.0),warps=$warps, atomics=$atomics)) setup=(data=CUDA.ones($n))
        @show result

        # add result to results
        push!(results, result)
        push!(N, n)

        n = n * 2

        sleep(1)
    end 

    df_benchmark = mapreduce(vcat, zip(results,N)) do (x, y)
        df = DataFrame(x)
        df.N .= y
        df
    end

    #return df_benchmark
    return df_benchmark
end

function benchmark_CUDA(warps)

    results = []
    N = []
    
    n =128 
    while n < 5_000_000 

        # this will take longer as every iteration the function will be parsed
        result = @benchmark CUDA.@sync( GPUArrays.mapreducedim!(x->x, *, similar(data,(1)), data; init=Float32(0.00),warps=$warps)) setup=(data=CUDA.ones($n))
        @show result

        # add result to results
        push!(results, result)
        push!(N, n)

        n = n * 2

        sleep(1)
    end 

    df_benchmark = mapreduce(vcat, zip(results,N)) do (x, y)
        df = DataFrame(x)
        df.N .= y
        df
    end

    #return df_benchmark
    return combine(groupby(df_benchmark, :N), 
                  :times => minimum => :minimum,
                  :times => mean => :average,
                  :times => median => :median,
                  :times => maximum => :maximum,
                  :allocs => first => :allocations,
                  :memory => first => :memory_estimate)
end


# read CSV
function benchmark_NVIDIA()

    ka = benchmark_KA(false, false)
    ka_warps = benchmark_KA(false, true)
    ka_atomics = benchmark_KA(true, false)
    ka_warps_atomics = benchmark_KA(true, true)

    cuda_jl_warps = benchmark_CUDA(true)
    cuda_jl = benchmark_CUDA(false)
    
    cuda_jl_warps.warps .= true
    cuda_jl_warps.atomics .= false

    cuda_jl.warps .= false
    cuda_jl.atomics .= false

    ka.warps .= false
    ka.atomics .= false
    
    ka_warps.warps .= true
    ka_warps.atomics .= false
    
    ka_atomics.warps .= false
    ka_atomics.atomics .= true
    
    ka_warps_atomics.warps .= true
    ka_warps_atomics.atomics .= true
    
    # combine all ka dataframes
    ka = vcat(ka, ka_warps, ka_atomics, ka_warps_atomics)
    ka.package .= "KernelAbstractions.jl"
    ka.filter .= false
    # combine all cuda jl dataframes
    cuda_jl = vcat(cuda_jl_warps, cuda_jl)
    cuda_jl.package .= "CUDA.jl"
    cuda_jl.filter .= false
    
    # combine all dataframes
    df = vcat(ka, cuda_jl)

    #write df to nvidia.csv

    CSV.write("nvidia.csv", df)
end

function benchmark_APPLE()
    metal_jl_warps = benchmark(Metal, :(GPUArrays.mapreducedim!(x->x, +, similar(data,(1)), data; init=Float32(0.0), shuffle=true)); filtered=false)    
    metal_jl = benchmark(Metal, :(GPUArrays.mapreducedim!(x->x, *, similar(data,(1)), data; init=Float32(0.0))); filtered=false)
    
    ka = benchmark(Metal, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0))); filtered=false)
    ka_warps = benchmark(Metal, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0), warps=true)); filtered=false)
    #ka_atomics = benchmark(Metal, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0), atomics=true )); filtered=false)
    #ka_warps_atomics = benchmark(Metal, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0), warps=true, atomics=true)); filtered=false)
    
    metal_jl_warps.warps .= true
    metal_jl_warps.atomics .= false

    metal_jl.warps .= false
    metal_jl.atomics .= false

    ka.warps .= false
    ka.atomics .= false
    
    ka_warps.warps .= true
    ka_warps.atomics .= false
    
    #ka_atomics.warps .= false
    #ka_atomics.atomics .= true
    
    #ka_warps_atomics.warps .= true
    #ka_warps_atomics.atomics .= true
    
    # combine all ka dataframes
    #ka = vcat(ka, ka_warps, ka_atomics, ka_warps_atomics)
    ka = vcat(ka, ka_warps)
    ka.package .= "KernelAbstractions.jl"
    ka.filter .= false
    # combine all cuda jl dataframes
    metal_jl = vcat(metal_jl_warps, metal_jl)
    metal_jl.package .= "Metal.jl"
    metal_jl.filter .= false
    
    # combine all dataframes
    df = vcat(ka, metal_jl)

    #write df to nvidia.csv

    CSV.write("apple.csv", df)
end

#benchmark_NVIDIA()
NVTX.enable_gc_hooks()

result = @benchmarkable CUDA.@sync( begin mapreducedim(x->x, *, tmp, data, partial; init=Float32(0.00)) 
end ) setup=(begin
data=CUDA.ones(2^22) 
partial = CUDA.ones(68)
tmp = similar(data,(1))
CUDA.reclaim()
end) evals=10 samples=10000 seconds = 100

display(run(result))



