# import required library

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create dataframe from image and additional information
def create_new_dataset(location):
  filename,freshness,fruit=[],[],[]
  for file in tqdm(os.listdir(location)):
    for img in os.listdir(os.path.join(location,file)):
       freshness.append(1 if file[0]=='f' else 0)
       fruit.append(file[5:] if file[0]=='f' else file[6:])
       filename.append(os.path.join(location,file,img))
  df=pd.DataFrame({
      "filename":filename,
      "freshness":freshness,
      "fruit":fruit
  })
  return df
train_dataset=create_new_dataset('/content/dataset/Train')

test_dataset=create_new_dataset('/content/dataset/Test')

df=pd.concat([train_dataset,test_dataset],axis=0)

"""balancing dataset(In dataset some fruit item number is 
 more then 2000 and some less then 500 )"""

counts = df['fruit'].value_counts()
df_new = pd.DataFrame(columns= ['filename', 'fruit', 'freshness'])

for (key, value) in counts.items():
    if value > 1500:
        df_temp = df[df['fruit'] == key].sample(n = 1500)
    else:
        df_temp = df[df['fruit'] == key]

    df_new = pd.concat([df_new, df_temp], axis = 0)

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df_new['fruit_labeling']=LabelEncoder().fit_transform(df_new['fruit'])

label_to_fruit = df_new[['fruit_labeling', 'fruit']].drop_duplicates().sort_values('fruit_labeling')

df_new.drop('fruit', axis=1, inplace=True)
df_new['freshness'] = df_new['freshness'].astype(int)
#  split train and test
from sklearn.model_selection import train_test_split
train,test=train_test_split(df_new,test_size=0.2,random_state=42)

# generate new image based on available image
train_gendata=ImageDataGenerator(
     rescale=1./255,
     rotation_range=35,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True
)
test_gendata=ImageDataGenerator(rescale=1./225)


train_generator=train_gendata.flow_from_dataframe(train ,
                                                  directory = None,
                                                  x_col='filename',
                                                  y_col=['freshness','fruit_labeling'],
                                                  target_size=(200,200),
                                                  class_mode='raw'
                                                  )

test_generator=test_gendata.flow_from_dataframe(test ,
                                                  directory = None,
                                                   x_col='filename',
                                                   y_col=['freshness','fruit_labeling'],
                                                   target_size=(200,200),
                                                  class_mode='raw'
                                                  )

# Custom generator function
def custom_generator(generator):
    for x, y in generator:
        # Assuming y is a list of labels
        fruit_labeling= np.array([label[0] for label in y])
        freshness = np.array([label[1] for label in y])
        # Reshape freshness to (batch_size, 1) to match the model's output
        freshness = freshness.reshape((-1, 1))
        yield x, (fruit_labeling, freshness)

def dataset_from_generator(generator, output_signature):
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


output_signature = (
    tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),
    (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,1), dtype=tf.float32)
    )
)
train_gen = dataset_from_generator(lambda: custom_generator(train_generator), output_signature)
test_gen = dataset_from_generator(lambda: custom_generator(test_generator), output_signature)

# trian by pretrain model MobileNet
from tensorflow.keras.applications import MobileNet
from keras.layers import *
from keras.models import Model

mobilenet=MobileNet(include_top=False,input_shape=(200,200,3))
mobilenet.trainable=False
output=mobilenet.layers[-1].output

flatter = Flatten()(output)

# ANN for freshness
dense_f1 = Dense(512, activation='relu')(flatter)
dense_f2 = Dense(512, activation='relu')(dense_f1)
output_f = Dense(1, activation='sigmoid', name='freshness')(dense_f2)

# ANN for fruit labeling
dense_fsh1 = Dense(512, activation='relu')(flatter)
dense_fsh2 = Dense(512, activation='relu')(dense_fsh1)
output_fsh = Dense(11, activation='softmax', name='fruit_labeling')(dense_fsh2)
model = keras.Model(inputs=mobilenet.input, outputs=[output_f, output_fsh])

model.compile(optimizer='adam',loss={'freshness':'binary_crossentropy','fruit_labeling': 'sparse_categorical_crossentropy'},metrics={ 'freshness': 'accuracy','fruit_labeling': 'accuracy'} )
STEP = len(train) // train_generator.batch_size
VALSTEP = len(test) // test_generator.batch_size

model.fit(train_gen,epochs=5,batch_size=32,validation_data=test_gen,steps_per_epoch=STEP ,validation_steps=VALSTEP)

import pickle
with open(model,'rb') as f:
  model=pickle.load(f)
