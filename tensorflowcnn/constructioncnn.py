import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
import os

data = input_data.read_data_sets('data/fasion',one_hot=True)

print("training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("training set (labels) shape:{shape}".format(shape=data.train.labels.shape))

print("testing set (images) shape: {shape}".format(shape=data.test.images.shape))
print("testing set (labels) shape:{shape}".format(shape=data.test.labels.shape))

label_dict = { 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',}
plt.figure(figsize=[5,5])

plt.subplot(121)
curr_img = np.reshape(data.train.images[0],(28,28))
print("train labels:",data.train.labels)
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img,cmap='gray')
plt.title("(label:"+str(label_dict[curr_lbl])+ ")")

plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
print("curr_img:",curr_img)
print("test labels:",data.test.labels)
curr_lbl = np.argmax(data.test.labels[0,:])
print("curr_label",curr_lbl)
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")


data.train.images[0]

print(np.max(data.train.images[0]))
print(np.min(data.train.images[0]))

train_x = data.train.images.reshape(-1,28,28,1)
test_x = data.test.images.reshape(-1,28,28,1)
print(train_x.shape,test_x.shape)

train_y = data.train.labels
test_y = data.test.labels

print(train_y.shape,test_y.shape)

training_iters = 20
learning_rate=0.001
batch_size=128

n_input = 28
n_classes=10

x = tf.placeholder("float",[None,28,28,1])
y = tf.placeholder("float",[None,n_classes])

def conv2d(x,W,b,strides = 1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# intializing the weight and bias

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x,weights,biases):
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,k=2)

    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,k=2)

    conv3 = conv2d(conv2,weights['wc3'],biases['bc3'])
    conv3 = maxpool2d(conv3,k=2)


    fc1 = tf.reshape(conv3,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

    # Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1
    # In TensorFlow, you define both the activation and the cross entropy loss functions in one line. You pass two parameters which are the predicted output and the ground truth label y. You will then take the mean (reduce_mean) over all the batches to get a single loss/cost value

pred = conv_net(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()

    # Training and testing
    # launch tf graph-it will run all the operation
    # then run the session -execute the varibale and evaluate the tensor
    # define the loop that run no.of times iteration

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./output',sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_x)//batch_size):
            batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            print("batch_x:",batch_x.shape)
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            print("batch_y:",batch_y.shape)
            opt = sess.run(optimizer, feed_dict={x:batch_x,y:batch_y})
            loss,acc = sess.run([cost,accuracy], feed_dict={x:batch_x,y:batch_y})

        print("iter"+str(i)+",loss="+":.6f".format(loss) +",Training accuracy="+"{:.5f}".format(acc))
        print("optimization finished")

        test_acc,valid_loss = sess.run([accuracy,cost],feed_dict={x:test_x,y:test_y})


        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)

        print("testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()
