using DataFrames, Tables, Statistics, CSV
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000
BenchmarkTools.DEFAULT_PARAMETERS.evals = 10

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
