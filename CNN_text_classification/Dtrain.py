import data_helper

def lis(x):
    num = 0
    re = []
    for a in x:
        for b in a:
            re.append(b)
    return re


def chan(x):
    return [[a] for a in x]

x_inp,y_input = data_helper.load_data()

x_input = [lis(a) for a in x_inp]


import tensorflow as tf
print len(x_input)
print len(y_input[0])

print"# Network Parameters"
# Network Parameters

dropout = 0.5 # Dropout, probability to keep units

the_text_length = 56
n_input = 300 * the_text_length
n_class = 2
the_out_d = 3
windows_h = 4
windows_l = 300
the_pool_k = 5


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32)

print " Parameters"
# Parameters
learning_rate = 0.5
training_iters = 100000
batch_size = 64
display_step = 20

filter_num = 270


def conv2d(img, w, b):
    cov =  tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], 
                                                  padding='VALID'),b))
    return tf.nn.relu(cov)


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

print "def conv_net"

def conv_net(_X, _weights, _biases, _dropout,filter_num):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, the_text_length, 300, 1])

    
    print "Convolution Layer"
    
    conves = []
    
    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k = 54)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)
    conves.append(conv1)
    
    
    conv2 = conv2d(_X, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k = 53)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)
    conves.append(conv2)
    
    
    conv3 = conv2d(_X, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = max_pool(conv3, k = 52)
    # Apply Dropout
    conv3 = tf.nn.dropout(conv3, _dropout)
    conves.append(conv3)
    
    
    
    
    num_filters_total = 3 * filter_num
    h_pool = tf.concat(3,conves)
    h_pool_flat = tf.reshape(h_pool, [-1,num_filters_total])
    h_drop = tf.nn.dropout(h_pool_flat,_dropout)
    
    
    
    
    print " Fully connected layer"
    out = tf.nn.xw_plus_b(h_drop, weights['out'], biases['out'], name="scores")
    
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([3, windows_l , 1, filter_num], mean=0.0, stddev=1.0)),
    
    'wc2': tf.Variable(tf.random_normal([4, windows_l , 1, filter_num], mean=0.0, stddev=1.0)),
    
    'wc3': tf.Variable(tf.random_normal([5, windows_l , 1, filter_num], mean=0.0, stddev=1.0)),
    
    'out': tf.Variable(tf.random_normal([3 * filter_num, n_class], mean=0.0, stddev=1.0))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([filter_num], mean=0.0, stddev=1.0)),
    'bc2': tf.Variable(tf.random_normal([filter_num], mean=0.0, stddev=1.0)),
    'bc3': tf.Variable(tf.random_normal([filter_num], mean=0.0, stddev=1.0)),
    'out': tf.Variable(tf.random_normal([n_class], mean=0.0, stddev=1.0))
}

print "# Construct model"
# Construct model
pred = conv_net(x, weights, biases, keep_prob,filter_num)

print "Define loss and optimizer"
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print "print "
# Initializing the variables
init = tf.initialize_all_variables()




print "Launch the graph"
print len(x_inp)
ma = len(x_input)
# Launch the graph
batches = data_helper.batch_iter(list(zip(x_input, y_input)), 64, 200)















