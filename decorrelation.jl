using LinearAlgebra, Distributions, Statistics, StatsBase, Distributions

m = 100 
ϵ = 1e-2

X = hcat(collect(range(-1.0,1.0,length = m)) + rand(m),
         collect(range(-1.0,1.0,length = m)) + rand(m),
         rand(m),
         collect(range(-1.0,1.0,length = m)))   # intentionally ill-conditioned matrix

C = cov(X')

Xd = eigen(C)

Xd.vectors*diagm(Xd.values)*inv(Xd.vectors)
C

Xd.vectors'*C*Xd.vectors
Cu = Xd.vectors'*X

cor(Cu)


