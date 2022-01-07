import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

def movedata(file):
    s00 = []
    s01 = []
    f = open(file, 'r', encoding='utf-8')
    for lines in f:
        ls = lines.strip('\n').replace('、', '/').replace('?', '').split('/')
        if (ls[0] == ''):
            continue
        if ((ls[0][0] == '*') ^ (ls[0][0] == 'N') ^ (ls[0][1] == 'O')):
            continue
        for i in ls:
            listm = i.split(" ")
            while '' in listm:
                listm.remove('')
            li1 = np.array(listm)[-16:-14]
            li2 = np.array(listm)[0:12]
            s00.append(li1)
            s01.append(li2)
            # s1.append(i.split(" "))
    f.close()
    data = DataFrame(s01)
    data_n = np.array(data.loc[:, [1, 2, 8, 9, 10, 11]])
    data_n = np.c_[data_n, s00]

    data_n = np.array(data_n)
    data_n = data_n.astype(float)
    return data_n

# s=[]
# f = open('none.txt', 'r',encoding='utf-8')
# for lines in f:
#     ls = lines.strip('\n').replace('、', '/').replace('?', '').split('/')
#     for i in ls:
#         list_n=i.split(" ")
#         while '' in list_n:
#             list_n.remove('')
#         li=np.array(list_n)[-12:]
#         s.append(li)
#         # s.append(i.split(" "))
# f.close()
# data_n=np.array(s)
# data_n = data_n.astype(float)
data_n=movedata('none.txt')
data_m1=movedata('move1.txt')
data_m2=movedata('move2.txt')
data_m3=movedata('move3.txt')
data_m4=movedata('move4.txt')
data_m5=movedata('move5.txt')


# print(data_n.shape)
# print(data_m.shape)

loc=np.mean(data_n,0)   #求矩阵各个参数的均值
scale=np.std(data_n,0)  #求无人矩阵各个参数的标准差*(10%、20%...)
# matrix_random=np.array(np.random.normal(loc,scale,size=(20020,8)))
#生成(20020,8)正态分布矩阵
matrix_random1=np.array(np.random.normal(loc[0],scale[0],size=(20020,1)))
matrix_random2=np.array(np.random.normal(loc[1],scale[1],size=(20020,1)))
matrix_random3=np.array(np.random.normal(loc[2],scale[2],size=(20020,1)))
matrix_random4=np.array(np.random.normal(loc[3],scale[3],size=(20020,1)))
matrix_random5=np.array(np.random.normal(loc[4],scale[4],size=(20020,1)))
matrix_random6=np.array(np.random.normal(loc[5],scale[5],size=(20020,1)))
matrix_random7=np.array(np.random.normal(loc[6],scale[6],size=(20020,1)))
matrix_random8=np.array(np.random.normal(loc[7],scale[7],size=(20020,1)))
matrix_random=np.c_[matrix_random1,
                    matrix_random2,
                    matrix_random3,
                    matrix_random4,
                    matrix_random5,
                    matrix_random6,
                    matrix_random7,
                    matrix_random8]

matrix_n=data_n

for i in range(1000):
    matrix_n=np.row_stack((matrix_n,data_n))

# move=matrix_random+data_m
move1=data_m1
move2=data_m2
move3=data_m3
move4=data_m4
move5=data_m5
none=matrix_random+matrix_n


moves1=[]
moves2=[]
moves3=[]
moves4=[]
moves5=[]
nones=[]
for lens in range(int(len(none)/20)):
    s0 = none[20 * lens:20 * (lens + 1), :]
    s1 = move1[20 * lens:20 * (lens + 1), :]
    s2 = move2[20 * lens:20 * (lens + 1), :]
    s3 = move3[20 * lens:20 * (lens + 1), :]
    s4 = move4[20 * lens:20 * (lens + 1), :]
    s5 = move5[20 * lens:20 * (lens + 1), :]

    s0 = np.array(s0).reshape(20, 8, 1)
    s1 = np.array(s1).reshape(20, 8, 1)
    s2 = np.array(s2).reshape(20, 8, 1)
    s3 = np.array(s3).reshape(20, 8, 1)
    s4 = np.array(s4).reshape(20, 8, 1)
    s5 = np.array(s5).reshape(20, 8, 1)

    moves1.append(s1)
    moves2.append(s2)
    moves3.append(s3)
    moves4.append(s4)
    moves5.append(s5)
    nones.append(s0)

y=[0 for i in range(len(nones))]
y1=[1 for i in range(len(moves1))]
y2=[2 for i in range(len(moves2))]
y3=[3 for i in range(len(moves3))]
y4=[4 for i in range(len(moves4))]
y5=[5 for i in range(len(moves5))]

X=nones+moves1+moves2+moves3+moves4+moves5
Y=y+y1+y2+y3+y4+y5
print(np.array(X).shape)
print(np.array(Y).shape)


cc = list(zip(X,Y))
import random
random.shuffle(cc)
X[:], Y[:] = zip(*cc)

count=int(0.8*len(X))
X_train=np.array(X[:count])
y_train=np.array(Y[:count])
X_test=np.array(X[count:])
y_test=np.array(Y[count:])


model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (1, 1), input_shape=(20, 8, 1), activation='relu'))
model.add(tf.keras.layers.Conv2D(16, (1, 1), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(32, (1, 1), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(32, (1, 1), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='softmax'))   #输出层  二分类问题，输出单元应该设置为1，激活函数使用sigmoid函数

model.summary()

model.compile(optimizer='adam',
             # loss='binary_crossentropy',
              loss='sparse_categorical_crossentropy',
             metrics=['acc'])    #二分类问题


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)#min_lr=0.000001
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# model_checkpoint = tf.keras.callbacks.ModelCheckpoint('./model/model_{epoch:02d}-{val_acc:.4f}.h5', save_best_only=True, save_weights_only=False)

histroy=model.fit(X_train, y_train, epochs=50,validation_data=(X_test, y_test), batch_size=64,callbacks=[reduce_lr])



score = model.evaluate(X_test, y_test, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
model.save('class.h5')

# list all data in history
print(histroy.history.keys())
# summarize history for accuracy
plt.plot(histroy.history['acc'])
plt.plot(histroy.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(histroy.history['loss'])
plt.plot(histroy.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#--------------------------------------------混淆矩阵------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# test_model = tf.keras.models.load_model('class.h5')
y_pred=model.predict_classes(X_test)


#labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = ['0','1', '2', '3', '4', '5']
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
# plt.savefig('../Data/confusion_matrix.png', format='png')
plt.show()
