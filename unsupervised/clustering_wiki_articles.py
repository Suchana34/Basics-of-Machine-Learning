# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)


# Perform the necessary imports
#TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays
#Combine knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia.
#DATASET -  https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/


from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components = 50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters = 6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)





# Import pandas
import pandas as pd

# Fit the pipeline to word-frequency array - articles
pipeline.fit(articles)

# Predict the cluster labels.
# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
# cluster of wikipedia articles formed



