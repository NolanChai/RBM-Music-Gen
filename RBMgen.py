"""
=== About the Restricted Boltzmann Machine === [1]

- Two-layered neural net
  * First layer [visible]
  * Second layer [hidden]

- Nodes are connected per consecutive layer

- No two nodes in a layer can interact with one another

- Deicisions to transmit information from one node to another is entirely random

=== Keywords [2]

- A SAMPLE may refer to individual training examples. A “batch_size” variable is hence the count of samples you sent to the neural network. That is, how many different 
  examples you feed at once to the neural network.
  
- TIMESTEPS are ticks of time. It is how long in time each of your samples is. For example, a sample can contain 128-time steps, where each time steps could be a 30th 
  of a second for signal processing. In Natural Language Processing (NLP), a time step may be associated with a character, a word, or a sentence, depending on the setup.
  
- FEATURES are simply the number of dimensions we feed at each time steps. For example in NLP, a word could be represented by 300 features using word2vec. In the case of 
  signal processing, let’s pretend that your signal is 3D. That is, you have an X, a Y and a Z signal, such as an accelerometer’s measurements on each axis. This means 
  you would have 3 features sent at each time step for each sample.

"""

#===Modules===

import numpy as np       #Our scientific computing library                      ----- Type 'pip install numpy'      in separate block if requirement not already satisfied
import pandas as pd      #Data analytics library                                ----- Type 'pip install pandas'     in separate block if requirement not already satisfied
import tensorflow as tf  #Self explanatory lol                                  ----- Type 'pip install tensorflow' in separate block if requirement not already satisfied
from tqdm import tqdm    #Awesome loading bar                                   ----- Type 'pip install tqdm'       in separate block if requirement not already satisfied
import midi_manipulation #Helper library to generate some music                 ----- Type 'pip install git+https://github.com/vishnubob/python-midi@feature/python3'
################################################### [3]
# In order for this code to work, you need to place this file in the same 
# directory as the midi_manipulation.py file and the Pop_Music_Midi directory

import midi_manipulation

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs

songs = get_songs('Pop_Music_Midi') #These songs have already been converted from midi to msgpack
print ("{} songs processed".format(len(songs)))
###################################################
#===Hyperparameters===

#---------------------------MIDI

lower = midi_manipulation.lowerBound
upper = midi_manipulation.upperBound
bound = upper - lower

timesteps = 15                                                                 #----- Referring to the number of timesteps

visible_neurons = 2 * bound * timesteps                                        #----- Size of the visible layer

hidden_neurons = 50

#---------------------------NNs

epochs = 200                                                                   #----- Referring to the number of epochs

batch = 100                                                                    #----- Batch size

learning_rate = tf.constant(0.005, tf.float32)

#Placeholder variable for storing data
data = tf.placeholder(tf.float32, [None, visible_neurons], name = "data")

#Matrix for storing weights
weights = tf.Variable(tf.random_normal([visible_neurons, hidden_neurons], 0.01), name = "weights")

#Bias vectors for the hidden layer and visible layer
bias_hidden = tf.Variable(tf.zeros([1, hidden_neurons], tf.float32, name = "bias_hidden"))
bias_visible = tf.Variable(tf.zeros([1, visible_neurons], tf.float32, name = "bias_visible"))

#Generative algorithm

data_sample = gibbs_sample(1) #Gibbs creates a sample from a multivariate probability distribution
#Each state depends on the previous state and randomness to take a sample across the distribution it creates. 

hidden_nodes = sample(tf.sigmoid(tf.matmul(data, weights) + bias_hidden))

hidden_sample = sample(tf.sigmoid(tf.matmul(hidden_nodes, weights) + bias_hidden))

#===Functions===

#This function lets us easily sample from a vector of probabilities
def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

#This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM

def gibbs_sample(k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, weights) + bias_hidden)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(weights)) + bias_visible)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, data_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), data])
    #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
    #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    data_sample = tf.stop_gradient(data_sample) 
    return data_sample
  
#------------------------------------------------
size_bt = tf.cast(tf.shape(data)[0], tf.float32)
weight_adder = tf.mul(learning_rate / size_bt, tf.subtract(tf.matmul(tf.transpose(data), hidden_nodes), tf.matmul(tf.transpose(data_sample), hidden_sample)))
visible_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(data, data_sample), 0, True))
hidden_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(hidden_nodes, hidden_sample), 0, True))
#------------------------------------------------
updt = [weights.assign_add(weight_adder), bias_visible.assign_add(visible_adder), bias_hidden.assign_add(hidden_adder)]
#------------------------------------------------
with tf.Session() as sess:
    #First, we train the model
    #initialize the variables of the model
    init = tf.global_variables_initializer()
    sess.run(init)
    #Run through all of the training data num_epochs times
    for epoch in tqdm(range(epochs)):
        for song in songs:
            #The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
            #Here we reshape the songs so that each training example is a vector with num_timesteps x 2*note_range elements
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0]/timesteps)*timesteps)]
            song = np.reshape(song, [song.shape[0]/timesteps, song.shape[1]*timesteps])
            #Train the RBM on batch_size examples at a time
            for i in range(1, len(song), batch_size): 
                tr_x = song[i:i+batch]
                sess.run(updt, feed_dict={x: tr_x})
    #Now the model is fully trained, so let's make some music! 
    #Run a gibbs chain where the visible nodes are initialized to 0
    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((50, visible_neurons))})
    for i in range(sample.shape[0]):
        if not any(sample[i,:]):
            continue
        #Here we reshape the vector to be time x notes, and then save the vector as a midi file
        S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
"""
=== Citations

[1] - https://skymind.ai/wiki/restricted-boltzmann-machine

[2] - https://stats.stackexchange.com/questions/264546/difference-between-samples-time-steps-and-features-in-neural-network ; Stack Overflow definitions

[3] - https://github.com/llSourcell/Music_Generator_Demo/blob/master/rbm_chords.py

"""
