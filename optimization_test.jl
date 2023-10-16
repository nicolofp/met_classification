# import Pkg
# Pkg.add("JuMP")
# Pkg.add("HiGHS")

# Test optimization model
# (integer solution)
using JuMP
using HiGHS

model = Model(HiGHS.Optimizer)
@variable(model, x >= 0, Int)             # Set the option of integer value
@variable(model, 0 <= y <= 30, Int)       # Set the option of integer value 
@objective(model, Min, 12x + 20y)
@constraint(model, c1, 6x + 8y >= 100)
@constraint(model, c2, 7x + 12y >= 120)
print(model)

optimize!(model)
value(x)
value(y)

# mu = 2.5
# println("On average, $(mu) units of thermal are used in the first stage.")

# Shortest path from A to E based on different costs 
G = [
    0 10 30 0 0
    0 0 20 0 0
    0 0 0 10 60
    0 15 0 0 50
    0 0 0 0 0
]
n = size(G)[1]
b = [1, 0, 0, 0, -1]
shortest_path = Model(HiGHS.Optimizer)
set_silent(shortest_path)
@variable(shortest_path, x[1:n, 1:n], Bin)
# Arcs with zero cost are not a part of the path as they do no exist
@constraint(shortest_path, [i = 1:n, j = 1:n; G[i, j] == 0], x[i, j] == 0)
# Flow conservation constraint
@constraint(shortest_path, [i = 1:n], sum(x[i, :]) - sum(x[:, i]) == b[i],)
@objective(shortest_path, Min, sum(G .* x))
optimize!(shortest_path)
objective_value(shortest_path)

value.(x)

# Minimum flow-cost problem - Network problem
c = [2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 38, 18]
b = [0, -19, 16, -33, 0, 0, 117]
G = [-1 -1  1  0  1  0  0  0  0  1  0  0;
      1  0  0  1  0  1  0  0  0  0  0  0;
      0  0 -1 -1  0  0  0  0  0  0  0  0;
      0  1  0  0  0  0  1  1  0  0  1  0;
      0  0  0  0 -1 -1 -1  0  1  0  0  0;
      0  0  0  0  0  0  0 -1 -1  0  0  1;
      0  0  0  0  0  0  0  0  0 -1 -1 -1]

lb = zeros(12)
ub = [10, 10, 10, 30, 100, 30, 30, 15, 15, 20, 20, 20]
n = size(G)[2]

network_balance = Model(HiGHS.Optimizer)
set_silent(network_balance)
@variable(network_balance, x[1:n])
# Arcs with zero cost are not a part of the path as they do no exist
@constraint(network_balance, G * x >= -b)
# Flow conservation constraint
@constraint(network_balance, x[1:n] >= lb)
@constraint(network_balance, x[1:n] <= ub)
@objective(network_balance, Min, sum(c .* x))
optimize!(network_balance)
objective_value(network_balance)

value.(x)
