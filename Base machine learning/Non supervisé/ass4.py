import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Ensure your dataset has a column named 'text' for the article content
data = pd.read_csv('your_dataset.csv')  # Replace with your dataset path

# Preprocess the text data (optional, depending on your dataset)
# You can use various NLP techniques here
# For example, lowercasing, removing punctuation, etc.

# Convert the text data into TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

# Apply NMF for matrix factorization
n_components = 5  # Set the number of components (topics/categories)
nmf_model = NMF(n_components=n_components, random_state=42)
W = nmf_model.fit_transform(tfidf_matrix)  # Document-topic matrix
H = nmf_model.components_  # Topic-term matrix

# Optionally, you can use K-means on the W matrix to cluster the articles
kmeans = KMeans(n_clusters=n_components, random_state=42)
kmeans.fit(W)
data['predicted_category'] = kmeans.labels_

# Display the results
print(data[['text', 'predicted_category']].head())

# Visualize the clusters (optional)
plt.figure(figsize=(10, 8))
sns.countplot(data['predicted_category'])
plt.title('Distribution of Predicted Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
