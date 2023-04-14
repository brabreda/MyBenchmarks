struct Config{
    THREADS_PER_WARP,         # size of warp 
    THREADS_PER_BLOCK,        # size of 
    NDRANGE,

    USE_ATOMICS,
    USE_LANES         
   }
end

@inline function Base.getproperty(conf::Type{Config{ THREADS_PER_WARP, THREADS_PER_BLOCK, NDRANGE, USE_ATOMICS, USE_LANES }}, sym::Symbol) where { THREADS_PER_WARP, NDRANGE, THREADS_PER_BLOCK, USE_ATOMICS, USE_LANES }
    if sym == :threads_per_warp
        THREADS_PER_WARP
    elseif sym == :threads_per_block
        THREADS_PER_BLOCK
    elseif sym == :ndrange
        NDRANGE
    elseif sym == :use_atomics
        USE_ATOMICS
    elseif sym == :use_lanes
        USE_LANES
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end