mutable struct MyGPUArray{T,N,B} <: AbstractGPUArray{T,N}
    storage::Union{Nothing,ArrayStorage{B}}
  
    maxsize::Int  # maximum data size; excluding any selector bytes
    offset::Int   # offset of the data in the buffer, in number of elements
  
    dims::Dims{N}
  
    function CuArray{T,N,B}(::UndefInitializer, dims::Dims{N}) where {T,N,B}
      check_eltype(T)
      maxsize = prod(dims) * sizeof(T)
      bufsize = if isbitstype(T)
        maxsize
      else # isbitsunion etc
        # type tag array past the data
        maxsize + prod(dims)
      end
      buf = alloc(B, bufsize)
      storage = ArrayStorage(buf, 1)
      obj = new{T,N,B}(storage, maxsize, 0, dims)
      finalizer(unsafe_finalize!, obj)
    end
  
    function CuArray{T,N}(storage::ArrayStorage{B}, dims::Dims{N};
                          maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,B}
      check_eltype(T)
      return new{T,N,B}(storage, maxsize, offset, dims)
    end
  end