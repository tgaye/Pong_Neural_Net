import tensorflow as tf
import cv2  # read in pixel data
import demo  # our class
import numpy as np  # math
import random  # random
from collections import deque  # queue data structure. fast appends. and pops. replay memory

# hyper params
ACTIONS = 3  # up,down, stay
# define our learning rate
GAMMA = 0.99
# for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
# store our experiences, the size of it
REPLAY_MEMORY = 500000
# batch size to train on
BATCH = 100


# create tensorflow graph
def createGraph():
    # convolutional layers and bias vectors (5 layers):
    # creates an empty tensor with all elements set to zero using given shape. (model needs a clean slate to begin training)
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Activation Function:
    # Computes rectified linear unit (RELU) on  a 2-D convolution given 4-D input and filter tensors.
    # each layer takes params from previous layer as input
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    # matrix multiplication gives us output tensor
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    # return input and output tensor
    return s, fc5


# deep q network. feed in pixel data to graph session
def trainGraph(inp, out, sess):
    # to calculate the argmax, we multiply the predicted output by vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])  # ground truth

    # cost optimization:
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices=1) # action
    cost = tf.reduce_mean(tf.square(action - gt)) # cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost) # optimization fucntion

    game = demo.PongGame() # initialize our game
    D = deque() # experience replay to store policies

    # get present frame / input
    frame = game.getPresentFrame() # intial frame
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY) # convert rgb to gray scale
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY) # binary colors, black or white
    inp_t = np.stack((frame, frame, frame, frame), axis=2) # stack frames, our input tensor

    # saver
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # training params
    t = 0
    epsilon = INITIAL_EPSILON

    # training loop
    while (1):
        # output tensor
        out_t = out.eval(feed_dict={inp: [inp_t]})[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame = game.getNextFrame(argmax_t)

        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))

        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        # add our input tensor, argmax tensor, reward and updated input tensor to experience replay
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # training iteration
        if t > OBSERVE:
            # get values from our replay memory
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # train on that
            train_step.run(feed_dict={
                gt: gt_batch,
                argmax: argmax_batch,
                inp: inp_batch
            })
        # update our input tensor the the next frame
        inp_t = inp_t1
        t = t + 1

        # print out where we are after saving
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step=t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t,
              "/ Q_MAX %e" % np.max(out_t))

def main():
    # create session
    sess = tf.InteractiveSession()
    # input layer and output layer by creating graph
    inp, out = createGraph()
    # train our graph on input and output with session variables
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()