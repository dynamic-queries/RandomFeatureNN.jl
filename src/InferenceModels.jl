abstract type AbstractInferenceModel end
abstract type AbstractSampler end
abstract type AbstractFeatureModel end

struct LinearFM <: AbstractFeatureModel end 
struct QuadraticFM <: AbstractFeatureModel end 

function (lfm::LinearFM)(σ::AbstractActivationFunction, rng, idx, ρ, L, x)
    d,m = size(x)
    idx_from, idx_to = idx
    l = length(idx_from)
    indices = wsample(rng, 1:l, ρ, L; replace=false, ordered=true)
    W = zeros(L,d)
    b = zeros(L)
    s1, s2 = get_weights(σ,lfm)
    for i=1:L
        temp = x[:, idx_from[indices[i]]]- x[:, idx_to[indices[i]]]
        W[i,:] .= s1*temp / (norm(temp) .+ 1e-12)^2
        b[i] .= s2 + vec(W[i,:]*x[:,idx_from[indices[i]]])
    end
    return W,b
end

function (qfm::QuadraticFM)(σ::AbstractActivationFunction, rng, idx, ρ, L, x)
    d,m = size(x)
    idx_from, idx_to = idx
    l = length(idx_from)
    indices = wsample(rng, 1:l, ρ, L; replace=false, ordered=true)
    k = floor(Int, d*(d-1)/2)
    W = zeros(L, k)
    b = zeros(L)
    s1, s2 = get_weights(σ, lfm)
    for i=1:L
        temp = kron(x[:, idx_from[indices[i]]],x[:, idx_from[indices[i]]])[1:k] - kron(x[:, idx_to[indices[i]]],x[:, idx_to[indices[i]]])[1:k]
        W[i,:] .= s1*temp / (norm(temp) .+ 1e-12)^2
        b[i] .= s2 + vec(W[i,:]*kron(x[:,idx_from[indices[i]]],x[:,idx_from[indices[i]]])[1:k])
    end
    return W,b
end

struct UniformSampler <: AbstractSampler end 
struct FiniteDifferenceSampler <: AbstractSampler end
struct DerivativeSampler <: AbstractSampler end 

function (us::UniformSampler)(rng, x, y, num_samples)
    d,m = size(x)
    idx_from = sample(rng, 1:m, num_samples, replace=false, ordered=true)
    idx_to = sample(rng, 1:m, num_samples, replace=false, ordered=true)
    ρ = (1/num_samples)*ones(num_samples)
    return (idx_from, idx_to), ρ
end

function (fds::FiniteDifferenceSampler)(rng, x, y, num_samples)
    d,m = size(x)
    idx_from = sample(rng, 1:m, num_samples, replace=false, ordered=true)
    idx_to = sample(rng, 1:m, num_samples, replace=false, ordered=true)
    ρ = zeros(num_samples)
    for i=1:num_samples
        ρ += norm(y[:,idx_from[i]]-y[:,idx_to[i]])/(norm(x[:,idx_from[i]]-x[:,idx_to[i]])+1e-12)
    end
    return (idx_from, idx_to), ρ/num_samples
end

function (ds::DerivativeSampler)(rng, x, y, num_samples)
    din,m = size(x)
    distances = zeros(m,m)
    for i=1:m
        for j=1:m
            distances[i,j] = norm(x[:,i]-x[:,j])
        end
    end
    dmax = maximum(distances)
    dthres = 0.25*dmax
    
    idx_from = []
    idx_to = []
    ρ = 0
    for i=1:m
        for j=1:m
            if distances[i,j] < dthres
                push!(idx_from,i)
                push!(idx_to,j)
                ρ += norm(y[:,i]-y[:,j])/(norm(x[:,i]-x[:,j]+1e-12))
            end 
        end 
    end 
    return (idx_from, idx_to), ρ/length(ρ)
end



struct RandomFeatureNN <: AbstractInferenceModel
    rng
    sampler::AbstractSampler
    feature_model::AbstractFeatureModel
    activation::AbstractActivationFunction
    N::Int
    regression_model::AbstractRegressionModel
    function RandomFeatureNN(model::AbstractFeatureModel, sampler::AbstractSampler, activation::AbstractActivationFunction, N::Int, reg::AbstractRegressionModel)
        new(Xoshiro(0), sampler, model, activation, N, reg)
    end 
end

function (rfm::RandomFeatureNN)(prob)
    @unpack xdata,ydata,atol,rtol = prob
    idx, ρ = rfm.sampler(rfm.rng, xdata, ydata, 5*rfm.N)
    W,b = rfm.feature_model(rfm.activation, rfm.rng, idx, ρ, rfm.N, xdata)
    F = rfm.activation.(W*xdata .- b)
    k = rfm.regression_model(F',ydata', rfm )
    f = x->k'*rfm.activation.(W*reshape(x,:,1) .- b)
return f