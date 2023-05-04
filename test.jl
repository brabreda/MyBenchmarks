using KernelAbstractions
using Metal


@kernel function test1(conf, b, a)
    threadIdx_global = @index(Global)

    a[threadIdx_global] = 15
end

@kernel function test2( b, a, conf)
    threadIdx_global = @index(Global)

    a[threadIdx_global] = 15
end

a = Metal.ones(10000)

backend = KernelAbstractions.get_backend(a)

partial = similar(a, 1,  1)

println(typeof(partial))

conf = KernelAbstractions.Config{
    KernelAbstractions.warpsize(backend),
    KernelAbstractions.maxgroupsize(backend),
    KernelAbstractions.maxconcurrency(backend),
    true,
    true 
}

test1(backend, 1024)(conf,partial,a, ndrange= 10000)
@show a[1]

test2(backend, 1024)(partial,a,conf, ndrange= 10000)
@show a[1]
