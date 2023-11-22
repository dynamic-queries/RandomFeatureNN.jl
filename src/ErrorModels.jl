abstract type AbstractErrorModel end

struct MAE <: AbstractErrorModel end
struct RMSE <: AbstractErrorModel end
struct RMAE <: AbstractErrorModel end
struct RRMSE <: AbstractErrorModel end

function (mae::MAE)(ypred, y)
    err = y .- ypred
    s = sum(err,2)
    return s./size(ypred,2)
end

function (rmse::RMSE)(ypred, y)
    err = (y .- ypred).^2
    s = sum(err,2)/size(ypred,2)
    return sqrt.(s)
end

function (rmae::RMAE)(ypred, y)
    err = y .- ypred
    s = sum(err,2)/size(ypred,2)
    ymin = minimum(y)
    return s ./ (ymin + 1e-12)
end

function (rrmse::RRMSE)(ypred, y)
    err = y .- ypred
    s = sum(err,2)/size(ypred,2)
    ymin = minimum(y)
    return sqrt.(s) ./ (ymin + 1e-12)
end

mutable struct ErrorLogger
    logs::Matrix

    function ErrorLogger(model, f, x, y)
        d_out,m = size(y)
        logs = model(reshape(f(x),d_out,m),y)
        new(logs)
    end 
end