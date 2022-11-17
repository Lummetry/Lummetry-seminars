from keras.datasets import mnist
from keras.models import Model
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras.backend as K

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test  = X_test / 255

X_train = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_test  = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

NR_NOISE_FEATS = 100
nr_classes = y_train.max() + 1
PIX_LABEL = nr_classes

tf_eval_input = layers.Input(shape=(X_train.shape[1],), dtype=np.float32)
tf_X = layers.Dense(units=1024, activation='relu')(tf_eval_input)
tf_X = layers.Dropout(0.3)(tf_X)
tf_X = layers.Dense(units=512, activation='relu')(tf_X)
tf_X = layers.Dropout(0.3)(tf_X)
tf_X = layers.Dense(units=256, activation='relu')(tf_X)
tf_readout = layers.Dense(units=nr_classes+1, activation='softmax')(tf_X)
evaluator = Model(inputs=tf_eval_input, outputs=tf_readout)
evaluator.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
evaluator_blocat = Model(inputs=tf_eval_input, outputs=tf_readout)
evaluator_blocat.trainable = False

tf_input_pictor_noise = layers.Input(shape=(NR_NOISE_FEATS,), name='inp_pictor_noise')
tf_input_pictor_label = layers.Input(shape=(1,), name='inp_pictor_label', dtype=np.int32)
EmbImage = layers.Embedding(input_dim=nr_classes, output_dim=5)
SqueezeLambdaLayer = layers.Lambda(lambda x: K.squeeze(x, axis=1))
tf_emb_image = EmbImage(tf_input_pictor_label)
tf_emb_image = SqueezeLambdaLayer(tf_emb_image)
tf_feats = layers.concatenate([tf_input_pictor_noise, tf_emb_image])
tf_X = layers.Dense(units=256, activation='relu')(tf_feats)
tf_X = layers.Dense(units=512, activation='relu')(tf_X)
tf_X = layers.Dense(units=1024, activation='relu')(tf_X)
tf_X = layers.Dense(units=784,  activation='tanh')(tf_X)
pictor = Model(inputs=[tf_input_pictor_noise, tf_input_pictor_label], outputs=tf_X)

tf_input_scoala_noise = layers.Input(shape=(NR_NOISE_FEATS,), name='inp_scoala_noise')
tf_input_scoala_label = layers.Input(shape=(1,), name='inp_scoala_label', dtype=np.int32)
tf_imagine_pictor = pictor([tf_input_scoala_noise, tf_input_scoala_label])
rezultat_evaluare = evaluator_blocat(tf_imagine_pictor)
scoala = Model(inputs=[tf_input_scoala_noise, tf_input_scoala_label], outputs=rezultat_evaluare)
scoala.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


def pictor_sample(batch_size=1):
  return [np.random.normal(size=(batch_size,NR_NOISE_FEATS)),
          np.random.randint(0, PIX_LABEL, size=(batch_size,1)).astype(np.int32)]

def plot_generated_images(epoch, pictor, examples=100):
  noise, labels = pictor_sample(batch_size=examples)
  figsize, dim = (10,10), (10, 10)
  generated_images = pictor.predict([noise, labels])
  generated_images = generated_images.reshape(examples,28,28)
  plt.figure(figsize=figsize)
  for i in range(generated_images.shape[0]):
      plt.subplot(dim[0], dim[1], i+1)
      plt.text(0.5,-0.1, "Label {}".format(labels[i,0]), size=7, ha="center")
      plt.imshow(generated_images[i], interpolation='nearest')
      plt.axis('off')
  plt.tight_layout()
  plt.savefig('gan/results/sparse_crossentropy_generated_image_ep_%d.png' % (epoch+1))
  return

nr_epochs = 20
batch_size = 32
nr_batches = X_train.shape[0] // batch_size
for i in range(nr_epochs):
  for step in tqdm(range(nr_batches)):
    start = step * batch_size
    end = (step + 1) * batch_size
    
    X_batch_true = X_train[start:end]
    y_batch_true = y_train[start:end]
    
    X_batch_false = pictor.predict(pictor_sample(batch_size=batch_size))
    y_batch_false = np.ones((batch_size,1)) * PIX_LABEL

    X_batch = np.concatenate((X_batch_true, X_batch_false))
    y_batch = np.concatenate((y_batch_true, y_batch_false))
  
    evaluator.train_on_batch(X_batch, y_batch)
    
    X_batch_noise, X_batch_labels = pictor_sample(batch_size=batch_size)
    y_batch_falsuri = X_batch_labels
    
    scoala.train_on_batch([X_batch_noise, X_batch_labels], y_batch_falsuri)
  #endfor
  if i == 0 or (i+1) % 10 == 0: plot_generated_images(i, pictor)
#endfor
