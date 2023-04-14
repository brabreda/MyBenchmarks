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


