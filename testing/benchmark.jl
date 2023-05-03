using GPUArrays, CUDA
using DataFrames, Tables, Statistics, CSV

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
function benchmark(backend, func::Expr; filtered=true)

    results = []
    N = []
    
    n =128 
    while n < 5_000_000 

        # this will take longer as every iteration the function will be parsed
        result = eval(Meta.parse("@benchmark $func setup=(x->x, +, data) setup=(data=$backend.rand($n))"))

        @show result

        # add result to results
        push!(results, result)
        push!(N, n)

        n = n * 2
    end 

    df_benchmark = mapreduce(vcat, zip(results,N)) do (x, y)
        df = DataFrame(x)
        df.N .= y
        df
    end

    if filter == true
        thresholds = Dict([(group.N[1], group.times[1]) for group in groupby(df_benchmark,:N)])
        #filter per N the rows with times < treshold
        df_benchmark = filter(row -> row.times < thresholds[row.N], df_benchmark)
        df_benchmark = filter(row -> row.times < 1_000_000, df_benchmark)
    end


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
    cuda_jl_warps = benchmark(CUDA, :(GPUArrays.mapreducedim!(x->x, +, similar(data,(1)), data; init=Float32(0.0), shuffle=true)); filtered=false)    
    cuda_jl = benchmark(CUDA, :(GPUArrays.mapreducedim!(x->x, +, similar(data,(1)), data; init=Float32(0.0))); filtered=false)
    
    ka = benchmark(CUDA, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0))); filtered=false)
    ka_warps = benchmark(CUDA, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0), warps=true)); filtered=false)
    ka_atomics = benchmark(CUDA, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0), atomics=true )); filtered=false)
    ka_warps_atomics = benchmark(CUDA, :(mapreducedim(x->x, +, similar(data,(1)), data; init=Float32(0.0), warps=true, atomics=true)); filtered=false)
    
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

benchmark_NVIDIA()




