using Profile, PProf
using CUDA, GPUArrays
using NVTX
using BenchmarkTools

Profile.init(;delay=0.000001)

path = "profiles"

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000
NVTX.enable_gc_hooks()

# create delete dir and its content 
if isdir(path)
    rm(path; recursive=true, force=true)
end

mkdir(path)

result = @benchmarkable CUDA.@sync( begin 
    @profile begin
        NVTX.@range "full reduction" begin 
            GPUArrays.mapreducedim!(x->x, *, partial ,data; init=Float32(0.00)) 
        end
    end
end) setup=(
    begin
        Profile.clear()
        data=CUDA.ones(2^22) 
        partial = CUDA.ones(1)
        CUDA.reclaim()
    end
) teardown = (
    begin
        files = length(readdir($path))
        pprof(;web=false,out="$path/$files.pb.gz")
    end
) evals=1 samples=10000 seconds = 10000

display(run(result))