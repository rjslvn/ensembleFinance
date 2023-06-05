import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url, header=0, index_col=0)[0]
symbols = table.index.tolist()

print (symbols)
