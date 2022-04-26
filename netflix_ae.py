# netflix_ae.py
# by Chan Gwak for DAT112 Neural Network Programming Assignment 2
# due 27 May 2021 Thursday

# A script that uses a non-variational autoencoder to predict the
# movie ratings of a user for all movies, given their existing
# movie ratings.
# Uses a modified version of the dataset for the Netflix Prize
# (https://netflixprize.com/).

# The file ratings.sparse.small.csv contains integer values (0-5)
# in rows representing users and columns representing movies.
# Values 1-5 represent user ratings, while 0 means that the user
# has not rated the movie.
# There are 10,000 users and 4499 movies.
# The first row contains movie labels X1, X2, ..., X4499.

# Note that no additional information (such as genre) is provided.
# It is entirely up to the model to discover patterns (e.g. a 
# certain user rates 5 for every movie, a movie is almost always 
# rated 3, etc.) through this unsupervised learning procedure.
# The only "goodness" that can be attributed to the model's
# prediction is how well it reproduces the existing ratings.

# If it has correctly identified the existing patterns, it should
# also reproduce the existing ratings in the testing data.

# Uses each user as a data point.
# A more sophisticated method would be to separate out user-movie
# pairs to be used as testing data, and to replace these with zeros
# in the remaining table which becomes training data. (melt, pivot)
# The results are satisfactory without this, however.



import pandas as pd
import numpy as np
import sklearn.model_selection as skms
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K


# Hyperparameters
train_set_size = 0.8 # train_size; Use 0.2 of data for testing
num_hidden = 900 # size of the dense hidden layer
latent_dim = 100 # size of the latent space after encoding
drop_rate = 0.8 # rate of dropout
num_iters = 25000
train_batch_size = 64 # batch_size for training
num_train_fx = 1 # number of times to train on artificial data in each iteration

# Note: num_iters should be at least 
#    (#epochs) * train_set_size * 10000 // train_batch_size ~ O(12500)

# Settings
verbose_step = 125 # how often model.predict should be verbose
save_models = False # whether to save the models


# Name of file containing the dataset.
# The file should be colocated with this script.
rati_fname = 'ratings.sparse.small.csv'

# Use the pandas package to read the file into a DataFrame.
print("\nReading netflix input file.")
rati_df = pd.read_csv(rati_fname)
_, num_movies = rati_df.shape #(10000, 4499)

# To faciliate operations, convert to a numpy array of floats.
rati = rati_df.to_numpy().astype(float)

# Split the data into training and testing data sets.
# (Shuffled by default pre-split.)
train_rati, test_rati = skms.train_test_split(rati, train_size=train_set_size)

# A function to build the autoencoder model.
def build_model(lat_dim, numnodes):

	# The model will be trained on (test_set_size * 10000) arrays, 
	# each of which has (num_movies) values containing ratings.
	input_rati = kl.Input(shape = (num_movies, ))

	##########################################################################
	# PART 1: Encoding

	# Encoding: Compress the (num_movies) values into
	# a smaller-dimensional latent space.
	x = kl.Dense(numnodes, activation = 'relu')(input_rati)
	x = kl.Dropout(drop_rate)(x)
	z = kl.Dense(lat_dim, activation = 'linear')(x)

	# Compose the encoder model.
	encoder_model = km.Model(inputs = input_rati, outputs = z)

	##########################################################################
	# PART 2: Decoding
	# The decoder input is the encoder's result (in the latent space).
	decoder_input = kl.Input(shape = (lat_dim, ))

	# Decoding: Take the values in the latent space and attempt to
	# retrieve the information they held before they were compressed.
	x = kl.Dense(numnodes, activation = 'relu')(decoder_input)
	x = kl.Dropout(drop_rate)(x)
	decoded = kl.Dense(num_movies, activation = 'relu')(x)

	# Compose the decoder model.
	decoder_model = km.Model(inputs = decoder_input, outputs = decoded)

	##########################################################################
	# PART 3: Complete model
	output_rati = decoder_model(encoder_model(input_rati))
	autoencoder_model = km.Model(inputs = input_rati, outputs = output_rati)

	##########################################################################

	return encoder_model, decoder_model, autoencoder_model


# Build an autoencoding neural network using Keras.
print("Building network.\n")
encoder, decoder, aenc = build_model(latent_dim, num_hidden)


# Zero-valued inputs are unrated movies, so they should not be
# used in the calculation of the cost function.
# Write a custom cost function, MMSE (Masked Mean Squared Error):
def mmse_loss(rati_true, rati_pred):

	# Both arguments are (num_movies,)-dim tensors.
	# The model should try to reproduce existing ratings (1-5),
	# but there is no restriction on what it predicts for
	# (user, movie) pairs without ratings.

	# Get a mask of [rati_true]'s zero values, and cast it as floats
	# such that 0.0 means zero value and 1.0 means nonzero.
	zero_mask = K.cast(K.not_equal(rati_true, 0.0), K.floatx())

	# Get the squared error, but use the mask to eliminate true zeros.
	masked_se = K.square((rati_true - rati_pred) * zero_mask)

	# Take the masked mean, using the sum of the mask as the total.
	return K.sum(masked_se) / K.sum(zero_mask)


print("\nCompiling model.")
aenc.compile(loss =  mmse_loss, optimizer = 'rmsprop')

print("Printing a summary of the model:\n")
aenc.summary()

# Train the network on the training data.
print("\nTraining network.\n")
verbosity = 0 # Initialize a verbosity for model.predict
for i in range(num_iters):
	
	if i % verbose_step == 0:
		print("Iteration number: ", i)
		verbosity = 1

	# Pick some random users/rows and get (real) their ratings.
	# (Set replace=False to avoid grabbing the same user twice in a single batch.)
	indices = np.random.choice(len(train_rati), train_batch_size, replace = False)
	r_ratings = train_rati[indices]

	# Train on these real ratings.
	fit = aenc.train_on_batch(r_ratings, r_ratings)

	for j in range(num_train_fx):

		# Use a different batch of users to create some fake ratings.
		indices = np.random.choice(len(train_rati), train_batch_size, replace = False)
		r_ratings = train_rati[indices]
		f_ratings = aenc.predict(r_ratings, 
			batch_size = train_batch_size, verbose = verbosity)

		# Train on the newly created fake ratings.
		# To avoid fixation on generated ratings, match to the original data.
		fit = aenc.train_on_batch(f_ratings, r_ratings)

	# Turn verbosity off
	verbosity = 0


# Print the final training accuracy.
score = aenc.evaluate(train_rati, train_rati)
print("The training score is:", score)

# Evaluate the network on the test data.
# Print the test accuracy.
score = aenc.evaluate(test_rati, test_rati)
print("The test score is:", score)

print("\nHere is a summary of the model again:\n")
aenc.summary()

# Save the final models.
if save_models:
	aenc.save('netflix_ae.h5')
	encoder.save('netflix_ae.h5')
	decoder.save('netflix_ae.h5')
