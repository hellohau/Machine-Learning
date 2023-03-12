import tensorflow as tf
import numpy as np

print(tf.__version__)

with open('harry.txt','r') as f:
    text = f.read()

vocab = sorted(set(text))

char_to_id = {char : id for id,char in enumerate(vocab)}

seq_len = 20
one_hots = []
for char in text:
    one_hot = np.zeros(len(vocab))
    one_hot[char_to_id[char]] = 1
    one_hots.append(one_hot)

input_data = []
output_data = []
for i in range(0,len(one_hots) - seq_len):
    input_data.append(one_hots[i:i+seq_len])
    output_data.append(one_hots[i+seq_len])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape = (seq_len,len(vocab)), return_sequences=True),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(48),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(vocab), activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

x = np.array(input_data)
y = np.array(output_data)

model.fit(x,y,epochs = 250,verbose=1,batch_size=128)

model.save("harry_lstm")