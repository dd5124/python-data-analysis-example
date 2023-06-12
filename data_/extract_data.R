library(bayesm)
library(readr)
data(orangeJuice)

typeof(orangeJuice)
sales_df <- orangeJuice$yx
store_df <- orangeJuice$storedemo

readr::write_csv(sales_df, "sales.csv")
readr::write_csv(store_df, "store.csv")

