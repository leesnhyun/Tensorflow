import tensorflow as tf
import matplotlib.pyplot as plt
import random
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# interactive plt on
plt.ion()

# params
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
dropout_rate = tf.placeholder("float")

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # MNIST data image size 28*28=784px (width, height)
y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes

# Xavier initialization
def xavier_init(n_inputs, n_outputs, uniform=True):
	if uniform:
		# 6 was used in the paper.
		init_range = tf.sqrt(6.0 / (n_inputs+n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
		return tf.truncated_normal_initializer(stddev=stddev)


# Store layers weight & bias
W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], initializer=xavier_init(256, 256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer=xavier_init(256, 256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer=xavier_init(256, 10))


B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))

# Construct model
_L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), B1)) # Hidden Layer with REUL activation
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2)) # Hidden Layer with REUL activation
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3)) # Hidden Layer with REUL activation
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4)) # Hidden Layer with REUL activation
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), B5) # No need to use softmax here


# Define loss func and optimizer( -tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(hypothesis, y) )
# with_logits => We didn't operate the softmax on the hypothesis
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
# AdamOptimizer -> the best alg so far

# Initializing the variables
init = tf.initialize_all_variables()
sess = tf.Session()

# Training cycle
sess.run(init)
	
for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(mnist.train.num_examples / batch_size)

	# Loop over all batches
	for i in range(total_batch):

		# Fit training using batch data
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		
		# Run optimization op(backprop) and cost op(to get loss value)
		_, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys, dropout_rate:0.7})
	
		# Compute average loss
		avg_cost += c / total_batch

	# Display logs per epoch step
	if (epoch+1) % display_step == 0:
		print "Epoch", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

print "Optimization Finished!"


# Get one add predict
r = random.randint(0, mnist.test.num_examples-1)
print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
print "Prediction: ", sess.run(tf.argmax(hypothesis, 1), {x: mnist.test.images[r:r+1], dropout_rate:1})

# Show the img
plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation="nearest")
plt.savefig('test.png')
plt.show()

# Test model
print '----------test----------'

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))

with sess.as_default() :
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels, dropout_rate:1})
