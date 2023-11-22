module RandomFeatureModels

    using LinearAlgebra
    using LinearSolve
    using Enzyme
    using UnPack
    using Statistics
    using StatsBase
    
    include("ActivationFunctions.jl")
    include("RegressionModels.jl")
    include("ErrorModels.jl")
    include("InferenceModels.jl")
    include("Interface.jl")
end