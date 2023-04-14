struct Config{

    THREADS_PER_WARP,          # size of warp 
    THREADS_PER_BLOCK,
    
    USE_ATOMICS     
   }
end

@inline function Base.getproperty(conf::Type{Config{ THREADS_PER_WARP, THREADS_PER_BLOCK, USE_ATOMICS}}, sym::Symbol) where { THREADS_PER_WARP, THREADS_PER_BLOCK, USE_ATOMICS}
    if sym == :threads_per_warp
        THREADS_PER_WARP
    elseif sym == :threads_per_block
        THREADS_PER_BLOCK
    elseif sym == :use_atomics
        USE_ATOMICS
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end




    
