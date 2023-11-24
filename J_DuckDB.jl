using DuckDB
using DataFrames
using Arrow

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
           FROM 'C:/Users/nicol/Downloads/Bioinformatics/GSE133758_globinblock_rawcounts.csv.gz'
           """));

DT = innerjoin(x, y, on = :target_id => :column0)

con = DBInterface.connect(DuckDB.DB, ":memory:")
@time x = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'C:/Users/nicol/Downloads/Bioinformatics/elegant/SRR12280931.fastq.gz-aligned-2/abundance.tsv'
           """));