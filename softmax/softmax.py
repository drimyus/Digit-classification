import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


learning_rate = 0.001
traning_epochs = 25
batch_size = 100
display_step = 1

num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float32, [None, num_classes])
W = tf.Variable(tf.zeros([num_input, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()

ckpt_path = "model/model_bin.ckpt"
saver = tf.train.Saver()

sess.run(init)

for epoch in range(traning_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

    if epoch % display_step == 0:
        # Calculate batch loss and accuracy
        print("Epochs= {}, Cost= {:.7f}".format(epoch + 1, avg_cost))
        saver.save(sess, ckpt_path)

print("Optimization Finished!")

correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
