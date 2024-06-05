import math
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

defalt_func=lambda x,a,b,c: (a*math.sin(b*x))+c



def visualize_data(*data):
    (X1,Y1),(X2,L,Y2)=data
    plt.subplot(2,2,1)
    plt.plot(X1,Y1)
    plt.title('train data')
    plt.subplot(2,2,2)
    plt.plot(X2,L)
    plt.title('test_data')
    plt.subplot(2,2,3)
    plt.plot(X2,Y2)
    plt.title('prediction_result')
    plt.subplot(2,2,4)
    plt.plot(X2,L)
    plt.plot(X2,Y2)
    plt.title('prediction_compare')
    plt.show()
    
    
def custom_func(data:np.ndarray,data_test:np.ndarray,functions:list=[defalt_func,],period:int=1000,func_scaler:bool=1):
    if func_scaler:
        temp=functions.copy()
        frequency=(data.size//period)*func_scaler
        for _ in range(frequency):
            functions=functions+temp
        
    c=random.uniform(-3,3)
    result=[]
    result_test=[]
    partition_size=data.size//len(functions)
    for x in range(len(data)):
        if (x%partition_size)==0 and functions:
            val = c if x==0 else func(data[x],a,b,c)
            func=random.choice(functions)
            functions.remove(func)
            a=random.uniform(-3,3)
            b=random.uniform(-3,3)
            c=val-func(data[x],a,b,0)
        result.append((data[x],func(data[x],a,b,c)))
        result_test.append((data_test[x],func(data_test[x],a,b,c)))
    return (result,result_test)
        
        

def data_generator(data_count:int=10000,funcs:list=[defalt_func,],data_range=(0,2*math.pi),distribution='uniform',scale:bool=False,func_scaler:bool=1,period:int=1000):
    
    if distribution=='uniform':
        data=np.random.uniform(data_range[0],data_range[1],data_count)
        data_test=np.random.uniform(2*data_range[0],2*data_range[1],data_count)
    elif distribution=='normal':
        data=np.random.normal(1.8,0.5,data_count) #have to change parameters by hand
        data_test=np.random.normal(1.8,0.5,data_count) #have to change parameters by hand
    else:
        raise ValueError('wrong distribution:  only acceptable distributions are "uniform" and "normal"')
    
    sorted_data=np.sort(data)
    sorted_data_test=np.sort(data_test)
    func_result,func_result_test=custom_func(sorted_data,sorted_data_test,funcs,func_scaler=func_scaler,period=period)
    sorted_label=list(zip(*func_result))[1]
    sorted_label_test=list(zip(*func_result_test))[1]
    random.shuffle(func_result)
    random.shuffle(func_result_test)
    data,label=list(zip(*func_result))
    data_test,label_test=list(zip(*func_result_test))
    label_range=(min(sorted_label),max(sorted_label))
    label_range_test=(min(sorted_label_test),max(sorted_label_test))
    
    sorted_label=np.array(sorted_label)
    data=np.array(data)
    label=np.array(label)
    sorted_label_test=np.array(sorted_label_test)
    data_test=np.array(data_test)
    label_test=np.array(label_test)
    if scale:
        scaler_data=MinMaxScaler((-1,1))
        scaler_label=MinMaxScaler((0,1))
        data=scaler_data.fit_transform(data.reshape(-1, 1))
        label=scaler_label.fit_transform(label.reshape(-1, 1))
        sorted_data=scaler_data.fit_transform(sorted_data.reshape(-1, 1))
        sorted_label=scaler_label.fit_transform(sorted_label.reshape(-1, 1))
        data_test=scaler_data.fit_transform(data_test.reshape(-1, 1))
        label_test=scaler_label.fit_transform(label_test.reshape(-1, 1))
        sorted_data_test=scaler_data.fit_transform(sorted_data_test.reshape(-1, 1))
        sorted_label_test=scaler_label.fit_transform(sorted_label_test.reshape(-1, 1))
    test_data_count=int(data_count*0.2)
    # result={'data':data,'sorted_data':sorted_data,
    #         'label':label,'sorted_label':sorted_label,
    #         'data_test':data_test[:test_data_count],'sorted_data_test':sorted_data_test[:test_data_count],
    #         'label_test':label_test[:test_data_count],'sorted_label_test':sorted_label_test[:test_data_count],
    #         'data_range':data_range,'label_range':label_range,
    #         'data_range_test':(data_range[0]*2,data_range[1]*2),'label_range_test':label_range_test,
    #         'is_scaled':scale}
    result={'data':data[test_data_count:],'sorted_data':sorted_data[test_data_count:],
        'label':label[test_data_count:],'sorted_label':sorted_label[test_data_count:],
        'data_test':data[:test_data_count],'sorted_data_test':sorted_data[:test_data_count],
        'label_test':label[:test_data_count],'sorted_label_test':sorted_label[:test_data_count],
        'data_range':data_range,'label_range':label_range,
        'data_range_test':data_range,'label_range_test':label_range,
        'is_scaled':scale}
    return result



functions={
    'lenear':lambda x,a,b,c: (a*x)+c ,
    'square':lambda x,a,b,c:a*(x**2)+b*x+c ,
    'sin':lambda x,a,b,c: (a*math.sin(b*x))+c ,
    'cos':lambda x,a,b,c: (a*math.cos(b*x))+c ,
    'square_root':lambda x,a,b,c:a*(math.sqrt(abs(b*x)))+c ,
    'reciprocal':lambda x,a,b,c:a*(1/(b*x))+c ,
    'logarithmic':lambda x,a,b,c:a*math.log2(abs(x*b)) +c
    
}

data_count=10000

cooked_data=data_generator(data_count=data_count,funcs=[functions['lenear'],functions['sin'],functions['cos'],functions['sin'],functions['square_root']],data_range=(-10,10),scale=False,func_scaler=0)
# cooked_data={'data':np.fromfile('data/data.bin'),'sorted_data':np.fromfile('data/sorted_data.bin'),
# 'label':np.fromfile('data/label.bin'),'sorted_label':np.fromfile('data/sorted_label.bin'),
# 'data_test':np.fromfile('data/data_test.bin'),'sorted_data_test':np.fromfile('data/sorted_data_test.bin'),
# 'label_test':np.fromfile('data/label_test.bin'),'sorted_label_test':np.fromfile('data/sorted_label_test.bin')}
# cooked_data['data'].tofile('data/data.bin')
# cooked_data['sorted_data'].tofile('data/sorted_data.bin')
# cooked_data['label'].tofile('data/label.bin')
# cooked_data['sorted_label'].tofile('data/sorted_label.bin')
# cooked_data['data_test'].tofile('data/data_test.bin')
# cooked_data['sorted_data_test'].tofile('data/sorted_data_test.bin')
# cooked_data['label_test'].tofile('data/label_test.bin')
# cooked_data['sorted_label_test'].tofile('data/sorted_label_test.bin')



#model created
model=Sequential([
    Input(shape=(1,)),
])
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate= 0.0001),loss='mean_squared_error',metrics=['accuracy'])
model.fit(x=cooked_data['data'],y=cooked_data['label'],batch_size=5,epochs=30,verbose=1)



temp=model.predict(x=cooked_data['data_test'],batch_size=5)
test_result=[]
i=0
for x in temp:
    test_result.append((cooked_data['data_test'][i],cooked_data['label_test'][i],x[0]))
    i+=1


test_data,test_label,test_result=list(zip(*sorted(test_result,key=lambda x:x[0])))
visualize_data((cooked_data['sorted_data'],cooked_data['sorted_label']),(test_data,test_label,test_result))


#notes