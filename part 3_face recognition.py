import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout,Conv2D,BatchNormalization,MaxPooling2D,Flatten,Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def face_model(train_data,test_data,classes_size=31,epochs=30,save_model:bool=0,load_model=0):
    if load_model:
            model=models.load_model('models/face.keras')
            accuracy=np.fromfile('models/face_accuracy.bin')
            test_accuracy=np.fromfile('models/face_val_accuracy.bin')
    else:
        model = Sequential([
        Input(shape=(32,32,3))])    
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3)) 
        model.add(Flatten())
        model.add(Dense(1024,activation='relu'))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(classes_size, activation='softmax'))


        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        train_info=model.fit(train_data,batch_size=32, epochs=epochs,verbose=1,validation_data=test_data)
        
        accuracy=np.array(train_info.history['accuracy'])
        test_accuracy=np.array(train_info.history['val_accuracy'])
    
    
        if save_model:
            model.save('models/face.keras')
            accuracy.tofile('models/face_accuracy.bin')
            test_accuracy.tofile('models/face_val_accuracy.bin')
        

    result={'model':model,'accuracy':accuracy,'test_accuracy':test_accuracy}
    return result


data=pd.read_csv('data\Dataset\Dataset.csv')
classes=data['label'].unique()
data['id']=data['id'].apply(lambda x: f'data/Dataset/Faces/{x}')
train_data=data[512:]
test_data=data[:512]

train_data=ImageDataGenerator(rescale=1./255).flow_from_dataframe(train_data,x_col='id',y_col='label',target_size=(32,32))
test_data=ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_data,x_col='id',y_col='label',target_size=(32,32))

model_results=face_model(train_data,test_data,epochs=100,load_model=1)

plt.plot(model_results['accuracy'])
plt.plot(model_results['test_accuracy'])
plt.legend(['accuracy', 'test accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.show()