
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils,get_file
from keras.optimizers import Adam
from model.CNN_LSTM import CNN
# from model.Vgg16 import CNN
# from model.Alex import create_base_network
# from model.Resnet50 import create_base_network
# from model.CNN import create_base_network
# from model.CNN_LSTM import CNN
import numpy as np
import utils

import cv2
from keras import backend as K
# K.set_image_dim_ordering('tf')

# WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
#                        'releases/download/v0.1/'
#                        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread((r".\data\image\train" + '/' + name))
            # img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        # X_train = utils.resize_image(X_train,(32,32))
        # X_train = X_train.reshape(-1,32,32,3)
        # Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 4)
        # yield (X_train, Y_train)
        X_train = utils.resize_image(X_train,(32,128))
        X_train = X_train.reshape(-1,32,128,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 4)
        yield (X_train, Y_train)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\train_3.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(9527)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.2)
    num_test = int(len(lines)*0.2)
    num_train = len(lines) - num_val - num_test
    # num_train = len(lines) - num_val

    # 建立VGG模型
    model = CNN()
    # model = CNN((32,128,3))
    # weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',WEIGHTS_PATH_NO_TOP,cache_subdir='models',file_hash='6d6bbae143d832006294945121d1f1fc')
    #
    # model.load_weights(weights_path,by_name=True)

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc',
                                    save_weights_only=False,
                                    save_best_only=True,
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc',
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )
    # trainable_layer = 19
    # for i in range(trainable_layer):
    #     model.layers[i].trainable = False

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-4),
            metrics = ['acc'])


    # 一次的训练集大小
    batch_size = 32

    # print('Train on {} samples, val on {} samples,test on {} samples, with batch size {}.'.format(num_train, num_val,num_test, batch_size))
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,batch_size))
    MC = ModelCheckpoint(filepath='./logs/better_one.h5', monitor='val_acc',
                         verbose=1,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='auto',
                         period=1)
    RL = ReduceLROnPlateau(monitor='val_acc',
                           factor=0.1,
                           patience=5,
                           verbose=1,
                           mode='auto',
                           cooldown=0,
                           min_lr=0)

    # 开始训练
    # model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=generate_arrays_from_file(lines[num_train:num_train+num_val], batch_size),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=10,
    #         initial_epoch=0,
    #         callbacks=[MC,RL])
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:num_train+num_val], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=30,
                        initial_epoch=0,
                        callbacks=[MC, RL])
    # model.save_weights(log_dir+'middle_one.h5')
    # for i in range(len(model.layers)):
    #     model.layers[i].trainable = True
    # # 交叉熵
    # model.compile(loss = 'categorical_crossentropy',
    #         optimizer = Adam(lr=1e-4),
    #         metrics = ['accuracy'])
    #
    # model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=6,
    #         initial_epoch=3,
    #         callbacks=[checkpoint_period1, reduce_lr])

    # model.save_weights(log_dir+'last_one.h5')
    model.load_weights(log_dir+'better_one.h5')
    # model.save(log_dir+'better_one.h5')
    loss, accuracy = model.evaluate_generator(generate_arrays_from_file(lines[num_train+num_val:], batch_size),steps=max(1, num_test//batch_size))
    print(loss, accuracy)

