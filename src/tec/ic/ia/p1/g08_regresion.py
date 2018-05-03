import numpy
import pandas
from tec.ic.ia.p1 import g08_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow as tf

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset




X,y = g08_data.shaped_data(10000)

# declare OneHotEncoder from sklearn
oneHot = OneHotEncoder()
# fit our X to oneHot encoder 
oneHot.fit(X)
# transform
X = oneHot.transform(X).toarray()
# fit our y to oneHot encoder




print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)

num_features = x_train.shape[1]

learning_rate = 0.01
training_epochs = 1000

tf.reset_default_graph()


learning_rate = 0.001
num_epochs = 1500
display_step = 300

# for visualize purpose in tensorboard we use tf.name_scope
with tf.name_scope("Declaring_placeholder"):    
    # X is placeholdre for iris features. We will feed data later on
    X = tf.placeholder(tf.float32, [None, len(x_train[0])] )
    # y is placeholder for iris labels. We will feed data later on
    y = tf.placeholder(tf.float32, [None, len(y_train[0])] )
    
with tf.name_scope("Declaring_variables"):
    # W is our weights. This will update during training time
    W = tf.Variable(tf.zeros([len(x_train[0]), len(y_train[0])]))
    # b is our bias. This will also update during training time
    b = tf.Variable(tf.zeros( [len(y_train[0])] ))










Z = tf.add(tf.matmul(X, W), b)
prediction = tf.nn.softmax(Z)

# Calculate the cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = y))

# Use Adam as optimization method
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

cost_history = numpy.empty(shape=[1],dtype=float)

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        _, c  = sess.run([optimizer, cost], feed_dict={X: x_train, y: y_train})
        #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \ "W=", sess.run(W), "b=", sess.run(b))
        cost_history = numpy.append(cost_history, c)
        
        
    # Calculate the correct predictions
    correct_prediction = tf.to_float(tf.greater(prediction, 0.5))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(y, correct_prediction)))


    prediction=tf.argmax(Z,1)
    print ("predictions", prediction.eval(feed_dict={X: x_test}, session=sess))


    print ("Train Accuracy:", accuracy.eval({X: x_train, y: y_train}))
    print ("Test Accuracy:", accuracy.eval({X: x_test, y: y_test}))
