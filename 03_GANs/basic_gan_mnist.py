from keras.datasets import mnist
from keras.models import Model
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test  = X_test / 255

X_train = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_test  = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

NR_NOISE_FEATS = 100

tf_eval_input = layers.Input(shape=(X_train.shape[1],), dtype=np.float32)
tf_X = layers.Dense(units=1024, activation='relu')(tf_eval_input)
tf_X = layers.Dropout(0.3)(tf_X)
tf_X = layers.Dense(units=512, activation='relu')(tf_X)
tf_X = layers.Dropout(0.3)(tf_X)
tf_X = layers.Dense(units=256, activation='relu')(tf_X)
tf_readout = layers.Dense(units=1, activation='sigmoid')(tf_X)
evaluator = Model(inputs=tf_eval_input, outputs=tf_readout)
evaluator.compile(optimizer='adam', loss='binary_crossentropy')
evaluator_blocat = Model(inputs=tf_eval_input, outputs=tf_readout)
evaluator_blocat.trainable = False

tf_pictor_input = layers.Input(shape=(NR_NOISE_FEATS,))
tf_X = layers.Dense(units=256, activation='relu')(tf_pictor_input)
tf_X = layers.Dense(units=512, activation='relu')(tf_X)
tf_X = layers.Dense(units=1024, activation='relu')(tf_X)
tf_X = layers.Dense(units=784,  activation='tanh')(tf_X)
pictor = Model(inputs=tf_pictor_input, outputs=tf_X)

tf_input_scoala = layers.Input(shape=(NR_NOISE_FEATS,))
tf_imagine_pictor = pictor(tf_input_scoala)
rezultat_evaluare = evaluator_blocat(tf_imagine_pictor)
scoala = Model(inputs=tf_input_scoala, outputs=rezultat_evaluare)
scoala.compile(optimizer='adam', loss='binary_crossentropy')


def plot_generated_images(epoch, pictor, examples=100):
  noise = np.random.normal(loc=0, scale=1, size=[examples, NR_NOISE_FEATS])
  figsize, dim = (10,10), (10, 10)
  generated_images = pictor.predict(noise)
  generated_images = generated_images.reshape(examples,28,28)
  plt.figure(figsize=figsize)
  for i in range(generated_images.shape[0]):
      plt.subplot(dim[0], dim[1], i+1)
      plt.imshow(generated_images[i], interpolation='nearest')
      plt.axis('off')
  plt.tight_layout()
  plt.savefig('gan/results/binary_crossentropy_generated_image_ep_%d.png' % (epoch+1))
  return

nr_epochs = 20
batch_size = 32
nr_batches = X_train.shape[0] // batch_size
for i in range(nr_epochs):
  for step in tqdm(range(nr_batches)):
    start = step * batch_size
    end = (step + 1) * batch_size
    
    X_batch_true = X_train[start:end]
    y_batch_true = np.ones((batch_size,1)) * 0.9
    
    X_batch_false = pictor.predict(np.random.normal(size=(batch_size, NR_NOISE_FEATS)))
    y_batch_false = np.zeros((batch_size,1))
    
    X_batch = np.concatenate((X_batch_true, X_batch_false))
    y_batch = np.concatenate((y_batch_true, y_batch_false))
  
    evaluator.train_on_batch(X_batch, y_batch)
    
    X_batch_imagini_false = np.random.normal(size=(batch_size, NR_NOISE_FEATS))
    y_batch_falsuri = np.ones((batch_size,1))
    
    scoala.train_on_batch(X_batch_imagini_false, y_batch_falsuri)
  #endfor
  if i == 0 or (i+1) % 10 == 0: plot_generated_images(i, pictor)
#endfor
