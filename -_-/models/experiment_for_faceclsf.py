import os
#create model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models



#dataset
''''

train_images, test_images =train_images/255.0, test_images/255.0

'''

#model

model = models.Sequential()
model.add(layers.Conv2D(512, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

checkpoint_path = "training_forcnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

early_stopper = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), callbacks=[early_stopper])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)