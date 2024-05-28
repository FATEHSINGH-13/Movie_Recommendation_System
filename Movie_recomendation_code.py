import pandas as pd  # To read the CSV file
import difflib  # To get close matches
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert texts into numeric values
from sklearn.metrics.pairwise import cosine_similarity  # To compare numeric values using cosine similarity

# Reading the CSV file
movie = pd.read_csv(r'C:/Users/fateh/Downloads/movies.csv')

# Select the columns from which we want to use as our keywords for recommendations
selected_columns = ['genres', 'keywords', 'cast', 'director']

# Removing the null values from the dataset by filling with empty strings
for column in selected_columns:
    movie[column] = movie[column].fillna('')

# Combining all selected columns into a single string for each movie
combine = movie['genres'] + ' ' + movie['keywords'] + ' ' + movie['cast'] + ' ' + movie['director']

# Converting the text to numeric values using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combine)

# Calculating cosine similarity between feature vectors
similarity = cosine_similarity(feature_vector)

# Taking the input from the user
movie_name = input("Enter the movie name: \n")

# Getting all movie titles as a list
title_list = movie['title'].tolist()

# Finding close matches to the input movie name
find_match = difflib.get_close_matches(movie_name, title_list)

# If no close match found, notify the user
if not find_match:
    print("No close match found for the movie. Please check the spelling or try a different movie.")
else:
    # The closest match to the user input
    close_match = find_match[0]

    # Getting the index of the closest match
    movie_index = movie[movie.title == close_match]['index'].values[0]

    # Creating a similarity score list
    similarity_score = list(enumerate(similarity[movie_index]))

    # Sorting the similarity scores from highest to lowest
    sorted_movie = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Printing the top 20 movie recommendations
    print("Movies Suggestions: \n")
    i = 1
    for movies in sorted_movie:
        index = movies[0]
        title_from_index = movie[movie.index == index]['title'].values[0]
        if i <= 20:
            print(i, title_from_index)
        i += 1
