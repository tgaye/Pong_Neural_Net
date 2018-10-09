# Pong_Neural_Net
Trained convolutional neural network that reads pixel data from game image and learns to beat CPU.  Practice with CNN's and machine 
learning with images.

### Game Creation (demo.py)
In this .py file we define variables needed to create our pong game window (L x W) as well as the dimensions of our paddles (player)
and the ball we play with.

This file imports 2 libraries, namely

'''
import pygame  # helps us make GUI games in python
import random  # help us define which direction the ball will start moving in
'''
After writing our functions for drawing the paddles, ball, and game window we are left with the task of writing the game logic.  How is
the ball going to bounce, where do our paddles start?

'''
class PongGame:
    def __init__(self):
        # random number for initial direction of ball
        num = random.randint(0, 9)
        
        # initialie positions of paddle
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        
        # and ball direction
        self.ballXDirection = 1
        self.ballYDirection = 1
        
        # starting point
        self.ballXPos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2
        
        # randomly decide where the ball will move
        if (0 < num < 3):
            self.ballXDirection = 1
            self.ballYDirection = 1
        if (3 <= num < 5):
            self.ballXDirection = -1
            self.ballYDirection = 1
        if (5 <= num < 8):
            self.ballXDirection = 1
            self.ballYDirection = -1
        if (8 <= num < 10):
            self.ballXDirection = -1
            self.ballYDirection = -1
'''

The remaining to functions are simply getPresentFrame() and getNextFrame() which allows us to update our game frame by frame as well
as pull our games current state.

# Beating Pong through Reinforcement Learning (Pong_RL.py)

In this .py file we begin with a function createGraph() which creates a tensorflow graph we will use for building our conv net.
I will be building a 5 layer network, although this choice is flexible and can be toyed with.

First we declare biases and weights for our 5 layers, heres an example of one:
'''
W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
b_conv1 = tf.Variable(tf.zeros([32]))
'''

We apply relu non-linear activation function to our first three convolutions:
'''
conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)
'''

Our output tensor is achieved through matrix multiplication of the 4th layer and weights of the 5th layer + bias of 5th layer.
'''
fc5 = tf.matmul(fc4, W_fc5) + b_fc5
'''

All that's left is to train our network!  This is done via trainGraph() which I'll give a brief rundown of.

The crucual part of this fuction is the cost optimization and choice of optimizer. I chose to go with AdamOptimizer after testing 
a few of the most popular.
'''
# cost optimization:
action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices=1) # action
cost = tf.reduce_mean(tf.square(action - gt)) # cost function
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost) # optimization fucntion
'''

The output tensor is rewarded if score of the user is positive (i.e they're scoring).
Our network starts out knowing nothing and will learn through iterations of gameplay which strategies work and which to avoid.

Mostly inspired by DeepMind.AI and their work solving games with machine learning, as well as an interest in the pygame library.







