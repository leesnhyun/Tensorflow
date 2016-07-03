import tensorflow as tf
import matplotlib.pyplot as plt
import random
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# interactive plt on
plt.ion()

# params
learning_rate = 0.8
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # MNIST data image size 28*28=784px (width, height)
y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes

# Set Model Weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) #Softmax (activation == hypothesis)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
		_, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})
	
		# Compute average loss
		avg_cost += c / total_batch

	# Display logs per epoch step
	if (epoch+1) % display_step == 0:
		print "Epoch", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

print "Optimization Finished!"


# Get one add predict
r = random.randint(0, mnist.test.num_examples-1)
print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
print "Prediction: ", sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r+1]})

# Show the img
plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation="nearest")
plt.savefig('test.png')
plt.show()

# Test model
print '----------test----------'

correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))

with sess.as_default() :
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
