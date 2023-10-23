using JuMP
using HiGHS
using GLPK
using DelimitedFiles

network_data_file = "C:/Users/nicol/Documents/Github_projects/met_classification/simple_net.csv"
network_data = readdlm(network_data_file, ',', header=true)
data = network_data[1]
header = network_data[2]
start_node = round.(Int64, data[:,1])
end_node = round.(Int64, data[:,2])
c = data[:,3]
u = data[:,4]

network_data2_file = "C:/Users/nicol/Documents/Github_projects/met_classification/simple_network_b.csv"
network_data2 = readdlm(network_data2_file, ',', header=true)
data2 = network_data2[1]
hearder2 = network_data2[2]
b = data2[:,2]

no_node = max( maximum(start_node), maximum(end_node) )
no_link = length(start_node)
nodes = 1:no_node
links = Tuple( (start_node[i], end_node[i]) for i in 1:no_link )

c_dict = Dict(links .=> c)
u_dict = Dict(links .=> u)

mcnf = Model(GLPK.Optimizer)
@variable(mcnf, 0<= x[link in links] <= u_dict[link])
@objective(mcnf, Min, sum(c_dict[link] * x[link] for link in links))
for i in nodes
    @constraint(mcnf, sum(x[(ii,j)] for (ii,j) in links if ii==i)
                    - sum(x[(j,ii)] for (j,ii) in links if ii==i) == b[i])
end
JuMP.optimize!(mcnf)

obj = JuMP.objective_value(mcnf)
x_star = JuMP.value.(x)

println("The optimal objective function value is = $obj")
println(x_star.data)