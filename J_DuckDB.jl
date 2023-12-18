using DuckDB
using DataFrames
using Arrow, HTTP


# https://envbran.s3.us-east-2.amazonaws.com/methylation/table1.arrow
df = DataFrame(Arrow.Table(download("https://envbran.s3.us-east-2.amazonaws" * 
                            ".com/methylation/table1.arrow")));

con = DBInterface.connect(DuckDB.DB, ":memory:")
#@time y = CSV.read("C:/Users/nicol/Downloads/Bioinformatics/elegant/SRR12280931.fastq.gz-aligned/abundance.tsv", DataFrame, delim = '\t')
@time x = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'C:/Users/nicol/Downloads/Bioinformatics/elegant/SRR12280931.fastq.gz-aligned-2/abundance.tsv'
           """));
x.target_id = string.(x.target_id)
for i in 1:size(x,1) 
    x.target_id[i] = split(x.target_id[i,1],"|")[1] 
end

# 

@time y = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'https://envbran.s3.us-east-2.amazonaws.com/methylation/table1.arrow'
           """));

DT = innerjoin(x, y, on = :target_id => :column0)

con = DBInterface.connect(DuckDB.DB, ":memory:")
@time x = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'C:/Users/nicol/Downloads/Bioinformatics/elegant/SRR12280931.fastq.gz-aligned-2/abundance.tsv'
           """));

mean = [2.,3.]
C = [0.2 0; 0 0.3]
d = MvNormal(mean, C)
x = rand(d, 2000)

td = MvTDist(5,mean, C)
y = rand(td,2000)