1. In een terminal ```ssh hydor-portforward``` uitvoeren
2. In nieuwe terminal:
    ````
    ssh hydor 
    ````
    en
    ````
    LD_LIBRARY_PATH=$(/home/bbreda/julia-1.8.5/bin/julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))') ncu --mode=launch /home/bbreda/julia-1.8.5/bin/julia
    ```` 
    en ```using CUDA```
3. Lokaal ncu-ui uitvoeren en connecteren via Connect (Activity: Interactive Profile, en bij Target Platform Attach en de goede PID kiezen).
4. Best auto profile aan zetten en Break on API error uitzetten via de toolbar in nsight compute. Dan op resume klikken.
Dan kan je een kernel uitvoeren in de Julia REPL en hij zou automatisch in je Nsight Compute GUI moeten verschijnen.


````
benchmark(CUDA, :(mapreduce(x->x, +, data)))
````

# testing van grote variantie
Bij het gebruik van --backtrace en --sample process-tree kwamen er wat errors opduiken bij het samplen. Dit kan eventueel worden opgelost door een lagere sample rate te gebruiken.
````
 LD_LIBRARY_PATH=$(/home/bbreda/julia-1.8.5/bin/julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))') /opt/nvidia/nsight-systems/2023.1.1/bin/nsys profile /home/bbreda/julia-1.8.5/bin/julia testing/CUDABenchmarkAndProfile.jl
````

Na het testen van CUDA.jl blijkt dat we hier met hetzelfde probleem zitten. zeer hoge variantie, die in lijn lift met die van mijn implementatie via KA.jl .


````
    port = 9000 + rand(1:1000)
    p = Tracy.capture("my_workload.tracy"; port)
    run(addenv(`$(Base.julia_cmd()) testing/CUDAtracytest.jl`,
               "TRACY_PORT" => string(port),
               "JULIA_WAIT_FOR_TRACY" => "1"))
    wait(p)
````