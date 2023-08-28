using GPUArrays
using NVTX
using DataFrames, Tables, Statistics, CSV
using CUDA
using BenchmarkTools
using Tracy

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000

Tracy.enable_tracepoint(CUDA,true)

result = @benchmarkable CUDA.@sync( Tracy.@tracepoint "reductie" begin GPUArrays.mapreducedim!(x->x, *, partial ,data; init=Float32(0.00)) 
end ) setup=(begin
data=CUDA.ones(2^22) 
partial = CUDA.ones(1)
CUDA.reclaim()
end) evals=10 samples=10000 seconds = 100

display(run(result))
