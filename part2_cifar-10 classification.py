import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense,Flatten,Input

def cifar_model(train_data,train_labels,test_data,test_labels,epochs=30,save_model:bool=0,load_model=0):
    if load_model:
            model=models.load_model('models/cifar.keras')
            accuracy=np.fromfile('models/cifar_accuracy.bin')
            test_accuracy=np.fromfile('models/cifar_val_accuracy.bin')
    else:
        model = Sequential([
        Input(shape=(32,32,3))])
        
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(16,activation='relu'))
        model.add(Dense(10, activation='softmax'))


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        train_info=model.fit(train_data, train_labels, batch_size=64, epochs=epochs,verbose=1,validation_data=(test_data, test_labels))
        
        accuracy=np.array(train_info.history['accuracy'])
        test_accuracy=np.array(train_info.history['val_accuracy'])
    
    
        if save_model:
            model.save('models/cifar.keras')
            accuracy.tofile('models/cifar_accuracy.bin')
            test_accuracy.tofile('models/cifar_val_accuracy.bin')
        

    result={'model':model,'accuracy':accuracy,'test_accuracy':test_accuracy}
    return result
        
    
    
    
(train_data, train_labels),(test_data, test_labels)=cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

train_data = (train_data.astype('float32'))/255
test_data = (test_data.astype('float32'))/255
 
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

result=cifar_model(train_data,train_labels,test_data,test_labels,epochs=30,load_model=1)
model=result['model']
accuracy=result['accuracy']
test_accuracy=result['test_accuracy']


plt.plot(accuracy)
plt.plot(test_accuracy)
plt.legend(['accuracy', 'test accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy')

predict_result = model.predict(test_data,batch_size=64)

classes = np.argmax(predict_result, axis=1)


X = plt.subplots(5, 5, figsize=(15,15))[1].ravel()

for i in np.arange(0, 25):
    X[i].imshow(test_data[i])
    X[i].set_title(f"label:{class_names[np.argmax(test_labels[i])]}\nprediction result:{class_names[classes[i]]}")
    X[i].axis('off')
    plt.subplots_adjust(wspace=1)
plt.show()