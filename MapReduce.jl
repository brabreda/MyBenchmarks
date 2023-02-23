using CUDA

function MyMapReduce(A)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    mem = @cuStaticSharedMem(Float64, 5)
    
    val = 0
    offset = 0
    @inbounds while idx + offset <= length(A)
        val = val + A[idx + offset]
        offset += gridDim().x * blockDim().x
    end

    mem[idx] = val
    sync_threads()

    val = 0
    if idx == 1
        for i in 1:10
            val += mem[i]
        end
    end
    
    return
end

function main()
    A = CUDA.rand(20)

    print(A)

    w = reduce(+, A)
    println(w)

    @cuda threads=5 blocks=2 MyMapReduce(A)
3
end

main()
