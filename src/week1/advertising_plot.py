import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline


data_filename  = '../data/row/Advertising.csv'

df = pd.read_csv(data_filename)

# Get a quick look of the data
df.iloc[0:2]

new_df = df.head(7)

print(new_df)

#mapeo de columnas a plot
plt.scatter(new_df['TV'],new_df['sales'])

# add etiquetas
plt.xlabel('TV budget')
plt.ylabel('Sales')

# add title
plt.title('TV Budget vs Sales')
plt.show()
