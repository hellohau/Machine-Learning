import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create the model
model = Sequential()

# Add the input layer and the first hidden layer with ReLU activation
model.add(Dense(8, input_dim=2, activation='relu'))

# Add the output layer with sigmoid activation
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print('Accuracy:', accuracy)
