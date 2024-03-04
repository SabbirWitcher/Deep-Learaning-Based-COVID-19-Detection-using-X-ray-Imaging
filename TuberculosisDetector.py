# Importing Necessary Modules

import numpy as np
from matplotlib import pyplot as plt
import keras
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


# Creating CNN Based Model in Keras

model = Sequential()
# Layer Block 1
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
# Layer Block 2
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Layer Block 3
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Layer Block 4
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Flattening the Matrix
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compiling the Model
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Printing the model summary
print("printing Summary:")
model.summary()

# Generating the Trainer

train_datagen=image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

# Training the Model

test_dataset = image.ImageDataGenerator(rescale=1./255)
print("Training Data: ")
train_generator = train_datagen.flow_from_directory(
    'Covid19andTuberculosis/Train',
    target_size = (224,224),
    batch_size = 5, # default 32
    class_mode = 'binary'
)
print(train_generator.class_indices)


# Generating the Validation Trainer
print("Validation Data: ")
validation_generator = test_dataset.flow_from_directory(
    'Covid19andTuberculosis/Val',
    target_size = (224,224),
    batch_size = 5, # 32 was default
    class_mode = 'binary'
)

# Fitting the Model
hist = model.fit(
    train_generator,
    steps_per_epoch = 200, # 32 gave a better result first
    epochs = 64, #First used 16
    validation_data = validation_generator,
    validation_steps = 16 # First used 16
)

# Saving the Model
model.save("model_adv2_tuberculosis.h5")

# Evaluating the Train Data
print("Evaluating Training Generator: ")
print("evaluating: ")
print(model.evaluate_generator(train_generator))
# Evaluating the Validation Data
print("Evaluating Test Generator: ")
print("evaluating: ")
print(model.evaluate_generator(validation_generator))

# Plotting The Model Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('Accuracy_tuberculosis.png')
# Plotting The Model Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('Loss_tuberculosis.png')





'''
***************************************************
*    THIS PART OF THE CODE IS KEPT HERE FOR       *
*        FUTURE USES OF CLASSIFICATION            *
***************************************************


import os
train_generator.class_indices

y_actual = []
y_test = []
for i in os.listdir("./Covid19/Val/Normal"):
  img = image.load_img("./Covid19/Val/Normal/"+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict(img)
  c=np.argmax(p,axis=1)
  y_test.append(c[0])
  y_actual.append(1)

for i in os.listdir("./Covid19/Val/Covid"):
  img = image.load_img("./Covid19/Val/Covid/"+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict(img)
  c=np.argmax(p,axis=1)
  y_test.append(c[0])
  y_actual.append(0)

y_actual = np.array(y_actual)
y_test = np.array(y_test)



# Sending New Data For Checking
import os
path = "./Normal/NORMAL2-IM-0130-0001.jpeg"
img = image.load_img(path, target_size=(224, 224))

img = image.img_to_array(img)/255
img = np.array([img])
print(img.shape)


predictions = (model.predict(img) > 0.5).astype("int32")
print('Classes:\n\n')
print(train_generator.class_indices)
if predictions==1:
    print("Normal")
else:
    print("Covid")



***********************************
*           END                   *
***********************************

'''
