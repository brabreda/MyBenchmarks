using GPUArrays
using NVTX
using DataFrames, Tables, Statistics, CSV
using CUDA
using BenchmarkTools
using Profile
using PProf

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000

NVTX.enable_gc_hooks()
Profile.clear()

result = @benchmarkable CUDA.@sync( begin GPUArrays.mapreducedim!(x->x, *, partial ,data; init=Float32(0.00)) 
end) setup=( begin
    data=CUDA.ones(2^22) 
    partial = CUDA.ones(1)
    CUDA.reclaim()
end) evals=10 samples=10000 seconds = 10000

display(run(result))
