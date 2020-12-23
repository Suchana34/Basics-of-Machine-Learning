#validating data includes various methods in pandas, removing and replacing np.nan and also arithmetic operations which can effect statistics of data 
# like mean, std, etc

# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace(8,np.nan,  inplace=True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())

# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())

#A variable that's computed from other variables is sometimes called a 'recode'