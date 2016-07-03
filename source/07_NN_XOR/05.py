import tensorflow as tf
import numpy as np

# DEEP NN for XOR
# +1 Hidden Layer
# using tensorboard

xy = np.loadtxt('train.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([5,4], -1.0, 1.0), name='Weight2')
W3 = tf.Variable(tf.random_uniform([4,1], -1.0, 1.0), name='Weight3')

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3") 

# TensorBoard histogram
w1_hist = tf.histogram_summary("weight_hs1", W1)
w2_hist = tf.histogram_summary("weight_hs2", W2)
w3_hist = tf.histogram_summary("weight_hs3", W3)
b1_hist = tf.histogram_summary("bias_hs1", b1)
b2_hist = tf.histogram_summary("bias_hs2", b2)
b3_hist = tf.histogram_summary("bias_hs3", b3)

x_hist = tf.histogram_summary("x", X)
y_hist = tf.histogram_summary("y", Y)


# Our Hypothesis
with tf.name_scope("layer2") as scope:
	L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer3") as scope:
	L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("layer4") as scope:
	hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# Cost function(Cross Entropy)
with tf.name_scope("cost") as scope:
	cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
	cost_summ = tf.scalar_summary("cost", cost)	

# Minimize
with tf.name_scope("train") as scope:
	a = tf.Variable(0.1) # Learning Rate, alpha
	optimizer = tf.train.GradientDescentOptimizer(a)
	train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session() as sess:
	sess.run(init)
	
	# tensorboard --logdir=./logs/xor_logs
	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph)
	
	# Fit the line(tensorboard->merge, always record)
	# for step in xrange(10000):
	#	summary, _ = sess.run([merged,train], feed_dict={X:x_data, Y:y_data})
	#	writer.add_summary(summary, step)

	# Test model
	correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accuracy_summ = tf.scalar_summary("accuracy", accuracy)

	
	# Fit the line.
	for step in xrange(10000):
		sess.run(train, feed_dict={X:x_data, Y:y_data})
		if step % 200 == 0:
			sess.run(accuracy, feed_dict={X:x_data, Y:y_data})
			summary,_ = sess.run([merged, train], feed_dict={X:x_data, Y:y_data})
			writer.add_summary(summary, step)

