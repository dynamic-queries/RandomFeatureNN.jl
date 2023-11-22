struct InferenceProblem 
    xdata::Matrix
    ydata::Matrix
    atol::Float64
    rtol::Float64
    
    function InferenceProblem(x,y; atol=1e-12,rtol=1e-12)
        new(x,y,atol,rtol)
    end 
end

mutable struct InferenceSolution
    prob::InferenceProblem
    f::Function
    ∇f::Function
    error_logs::Dict{String,ErrorLogger}
end 

function solve(prob::InferenceProblem, solver::AbstractInferenceModel, test_data::Tuple{Matrix}, em::AbstractErrorModel)
    f = solver(prob)
    Df = x->Enzyme.jacobian(Forward, f, x)
    train_error = ErrorLogger(em,f,prob.xdata,prob.ydata)
    test_error = ErrorLogger(em,f,test_data[1],test_data[2])
    logs = Dict("train"=>train_error, "test"=>test_error) 
    sol = InferenceSolution(prob, f, Df, logs)
    return sol
end 

# Mock test
x = reshape(LinRange(-2pi,2pi,100),1,:)
y = reshape(sin.(x),1,:)
xtest = reshape(LinRange(-pi,pi,100),1,:)
ytest = reshape(sin.(xtest),1,:)
atol = rtol = 1e-12
prob = InferenceProblem(x,y;atol=atol,rtol=rtol)
N = 100
reg = Tikhonov(1e-3)
solver = RandomFeatureNN(LinearFM(), UniformSampler(), Tanh(), N, reg)
error_model = MAE()
sol = solve(prob, solver, (xtest,ytest), error_model)