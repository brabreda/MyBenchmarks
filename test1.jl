macro macro1()
    quote
        println("macro1")
    end
end

macro macro2() return @macro1() end

@macro1()
@macro2()