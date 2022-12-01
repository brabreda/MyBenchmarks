using CUDA

function MyMapReduce(A)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    mem = @cuStaticSharedMem(Float64, 10)
    
    val = 0
    offset = 0

    """
    Eerst reduceren we op grote schaal we reduceren van de grote array naar een array die een 
    grote van gridDim().x * blockDim().x
    """


    @inbounds while idx + offset <= length(A)

        val = val + A[idx + offset]
        offset += gridDim().x * blockDim().x

    end

    @cuprintln(idx, " ", val)
    mem[idx] = val
    """
        probleem is hier dat shared memory enkel shared op basis van een block
    """
    sync_threads()

    if idx == 1
        for i in 1:10
            @cuprintln( mem[i])
        end
    end

    val = 0
    if idx == 1
        for i in 1:10
            val += mem[i]
        end
        @cuprintln(val)
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