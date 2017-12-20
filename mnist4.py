# -*- coding: utf-8 -*-
#TensorFlow实战才云科技6.4节例子
#Method:lenet-5
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# MNIST数据集相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
#配置神经网络参数
#第一层
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接节点数
FC_SIZE = 512

BATCH_SIZE = 100
LEARNING_RATE_BASE = 1e-4 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #正则化系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#网络前向传播结果一般都在这个函数计算
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #将第四层转化为第五层全连接输入格式（拉直），注意得到的第一维是batch
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    #只有全连接的权重加正则化,dropout也只在全连接用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight",[nodes,FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train !=None:fc1 = tf.nn.dropout(fc1,0.5)
    #这一层的输出再通过softmax就得到最后分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit

def train(mnist):
    train1 = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    # L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #这里不用滑动平均类，计算前向传播结果
    y = inference(x, train1, regularizer)
    #定义训练轮数变量，这个变量不可训练
    global_step = tf.Variable(0,trainable=False)
    #给定滑动平均衰减和训练轮数变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #所有变量使用滑动平均，除了global_step
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    #求当前batch交叉熵平均值
    cross_entropy_mean = tf.reduce_sum(cross_entropy)

    #总损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    #使用梯度下降优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step=global_step)
    #反向传播更新参数，滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')


    #检验结果是否正确
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #初始化会话窗口开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据
        validate_x, validate_y = mnist.validation.next_batch(BATCH_SIZE)
        validate_x = np.reshape(validate_x, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        validate_feed = {x:validate_x ,
                         y_:validate_y,
                         train1: None}

        #准备测试数据
        test_x, test_y = mnist.test.next_batch(BATCH_SIZE)
        test_x = np.reshape(test_x,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
        test_feed = {x:test_x,
                     y_:test_y,
                     train1: None}
        #迭代训练
        for i in range(TRAINING_STEPS):
            #每1000 输出一次结果
            if i % 1000 ==0:

                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print ("After %d training step(s),validation accuracy"
                       "using average model is %g ")%(i,validate_acc)

            #产生这一轮使用的一个batch训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
            sess.run(train_op,feed_dict={x:reshaped_xs,y_:ys, train1:1})

        #训练结束输出最后正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed,)
        print ("After %d training step(s),test accuracy using average"
               "model is %g "%(TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
