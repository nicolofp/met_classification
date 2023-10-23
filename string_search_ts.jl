using Statistics, Distributions, Random, Plots, StatsPlots 

rng = MersenneTwister(1234);
X = rand(rng,Normal(0,0.2),100,3000)
X = cumsum(X, dims = 2)

w = X[36,1234:1876] + rand(Normal(0,0.2),length(1234:1876))
N = length(1234:1876)

Z = reshape([[y,x]  for x=1:(size(X,2)-N+1), y=1:size(X,1)],(size(X,2)-N+1)*size(X,1))
Z = mapreduce(permutedims, vcat, Z)
Z = hcat(Z,Z[:,2] .+ (N-1))
Z = hcat(Z,zeros((size(X,2)-N+1)*size(X,1)))
@time for i in 1:size(Z,1)
    Z[i,4] = sum((w - X[Int(Z[i,1]),Int(Z[i,2]):Int(Z[i,3])]).^2)
end
    
Z = DataFrame(Z,:auto)
Z[Z.x4 .== minimum(Z.x4),:]