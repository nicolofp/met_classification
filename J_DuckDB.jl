using DuckDB
using DataFrames

con = DBInterface.connect(DuckDB.DB, ":memory:")

@time x = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'C:/Users/nicol/Documents/Datasets/Air_quality/waqi-covid19-airqualitydata-2023.csv'
           """));





