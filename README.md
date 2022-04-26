# netflix_ae.py

A script that uses a non-variational autoencoder to predict the movie ratings of a user for all movies, given their existing movie ratings. Uses a modified version of the dataset for the Netflix Prize (https://netflixprize.com/).

The file ratings.sparse.small.csv contains integer values (0-5) in rows representing users and columns representing movies. Values 1-5 represent user ratings, while 0 means that the user has not rated the movie. There are 10,000 users and 4499 movies. The first row contains movie labels X1, X2, ..., X4499.

Note that no additional information (such as genre) is provided. It is entirely up to the model to discover patterns through this unsupervised learning procedure. The only "goodness" that can be attributed to the model's prediction is how well it reproduces the existing ratings.

If it has correctly identified the existing patterns, it should also reproduce the existing ratings in the testing data.

Uses each user as a data point. A more sophisticated method would be to separate out user-movie pairs to be used as testing data, and to replace these with zeros in the remaining table which becomes training data (melt, pivot). The results are satisfactory without this, however.
