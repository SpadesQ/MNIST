# -*- coding: utf-8 -*-
#TensorFlow实战才云科技5.2节例子
#Method:三层全连接神经网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络参数
LAYER1_NODE = 500 #一个隐藏层

BATCH_SIZE = 100

LEARNING_RATE_BASE = 1e-4
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率

REGULARIZATION_RATE = 0.0001 #正则化系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#网络前向传播结果一般都在这个函数计算
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有使用滑动平均类，直接使用参数当前的取值。
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
        return tf.matmul(layer1,weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+
            avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    #隐藏层
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #输出层
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #这里不用滑动平均类，计算前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    #定义训练轮数变量，这个变量不可训练
    global_step = tf.Variable(0,trainable=False)
    #给定滑动平均衰减和训练轮数变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #所有变量使用滑动平均，除了global_step
    variable_averages_op = variable_averages.apply((tf.trainable_variables()))
    #计算使用了滑动平均的前向传播结果
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    #交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    #求当前batch交叉熵平均值
    cross_entropy_mean = tf.reduce_sum(cross_entropy)

    #L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)+regularizer(weights2)
    #总损失
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,

        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY)

    #使用梯度下降优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step=global_step)
    #反向传播更新参数，滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')


    #检验结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #初始化会话窗口开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}

        #准备测试数据
        test_feed = {x:mnist.test.images,
                     y_:mnist.test.labels}
        #迭代训练
        for i in range(TRAINING_STEPS):
            #每1000 输出一次结果
            if i % 1000 ==0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print ("After %d training step(s),validation accuracy"
                       "using average model is %g ")%(i,validate_acc)

            #产生这一轮使用的一个batch训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        #训练结束输出最后正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print ("After %d training step(s),test accuracy using average"
               "model is %g "%(TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
