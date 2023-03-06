````
LD_LIBRARY_PATH=$(/home/bbreda/julia-1.8.5/bin/julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))') ncu --mode=launch /home/bbreda/julia-1.8.5/bin/julia
````

````
ssh -L 49152:hydor:49152 bbreda@ssh3.elis.ugent.be -p 4444
````