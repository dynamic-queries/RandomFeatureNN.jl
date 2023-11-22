abstract type AbstractRegularization end

struct Tikhonov <: AbstractRegularization
    λ::Float64
end