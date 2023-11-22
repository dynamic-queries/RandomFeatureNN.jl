abstract type AbstractActivationFunction end

struct Tanh <: AbstractActivationFunction end
struct Relu <: AbstractActivationFunction end
struct Gelu <: AbstractActivationFunction end

function get_weights(σ::Tanh, lfm::LinearFM)
    return 2*log(1.5), log(1.5)
end 

function get_weights(σ::Union{Relu,Gelu}, lfm::LinearFM)
    return 1.0 ,0.0
end