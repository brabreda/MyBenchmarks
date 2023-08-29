struct Config{
    GROUPSIZE, 
    MAX_NDRANGE,

    ITEMS_PER_WORKITEM,
    USE_ATOMICS,
    USE_WARPS
    }

    function Config(groupsize, max_ndrange, items_per_workitem , use_atomics, use_warps)
        new{groupsize, max_ndrange, items_per_workitem, use_atomics, use_warps}()
    end
end

@inline function Base.getproperty(conf::Type{Config{GROUPSIZE, MAX_NDRANGE, ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS}}, sym::Symbol) where { GROUPSIZE, MAX_NDRANGE,ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS }
    println("here typed")
    println(sym)
    if sym == :groupsize
        GROUPSIZE
    elseif sym == :max_ndrange
        MAX_NDRANGE
    elseif sym == :items_per_workitem
        ITEMS_PER_WORKITEM
    elseif sym == :use_atomics
        USE_ATOMICS
    elseif sym == :use_warps
        USE_WARPS        
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end

@inline function Base.getproperty(conf::Type{Config{GROUPSIZE, MAX_NDRANGE, ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS}}, sym::Symbol) where { GROUPSIZE, MAX_NDRANGE,ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS }
    println("here normal")
    println(sym)
    if sym == :groupsize
        GROUPSIZE
    elseif sym == :max_ndrange
        MAX_NDRANGE
    elseif sym == :items_per_workitem
        ITEMS_PER_WORKITEM
    elseif sym == :use_atomics
        USE_ATOMICS
    elseif sym == :use_warps
        USE_WARPS        
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end

@generated function use_atomics(conf, x)
    println(conf.use_atomics)
    if conf.use_atomics
        quote
            
        end
    else 
        return quote
            x = x + 1

            return x
        end
    end
end

x=0
conf = Config(32, 1024, 4, false, false)

use_atomics(conf,x)

x = use_atomics(conf,x)

println(x)




