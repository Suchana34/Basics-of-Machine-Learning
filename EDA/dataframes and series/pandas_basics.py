#To get the number of rows and columns in a DataFrame, you can read its shape attribute.

#To get the column names, you can read the columns attribute. The result is an Index, which is a Pandas data structure that is similar to a list.

# Display the number of rows and columns
nsfg.shape

# Display the names of the columns
nsfg.columns

# Select column birthwgt_oz1: ounces
ounces = nsfg['birthwgt_oz1']

# Print the first 5 elements of ounces
print(ounces.head())