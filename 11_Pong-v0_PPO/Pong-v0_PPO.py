# %% [code]
#!usr/bin/env python3
#-*- coding: utf-8 -*-
# The bellow code is based in the code from "https://pylessons.com/PPO-reinforcement-learning/"

!pip install gym[atari] # Necessary to install the enviroment when using kaggle's kernel

# %% [code]
import tensorflow as tf
import keras as k
print(tf.__version__)
print(k.__version__)

# %% [code]
tf.test.is_gpu_available() # Check if GPU is available. Very good for find errors

# %% [code]
import os
import cv2
import gym
import random
import pylab
import numpy as np
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import threading
from threading import Thread, Lock
import time

from numba import cuda # Release GPU memory

K.clear_session() # reset the graphs and sessions
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.compat.v1.Session(config=config, graph=tf.compat.v1.get_default_graph())
sess.as_default()

# %% [code]
# Constants
STACK_SIZE = 4
FRAME_SIZE = (80,80)
MAX_EPISODES = 2000
BATCH_SIZE = 1000   # each one is a time step, not a episode. Change it if you have a GPU
STATE_SIZE = (STACK_SIZE, *FRAME_SIZE)

# %% [code]
class PGAgent(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = STATE_SIZE
        self.action_size = self.env.action_space.n
        self.lr = 0.000025
        self.gamma = 0.99
        self.epochs = 10
        self.max_average = -21.0 # First max_average Specific for Pong
        self.scores, self.episodes, self.average = [], [], []
        self.actor, self.critic = self.build_model()
        self.keep_running_thread = True
    
        self.save_path = 'models'
        self.path = 'PPO_env:{}_lr:{}'.format(self.env, self.lr)
        self.actor_name = os.path.join(self.save_path, self.path + '_actor')
        self.critic_name = os.path.join(self.save_path, self.path + '_critic')
        self.check_or_create_save_path()

        self.episode = 0 # used to track the episodes total count of episodes played through all environments
        self.lock = Lock() # lock all to update parameters without other thread interruption
        
        # threaded predict function
        # Initialize the graphs before using them in threads to avoid errros message
        self.actor._make_predict_function()
        self.actor._make_train_function()
        self.critic._make_predict_function()
        self.critic._make_train_function()
        
        self.session = tf.compat.v1.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()    
        self.graph.finalize()   # graph is not thread-safe, so you need to finilize it... Don't use gloabal graphs with thread


    def build_model(self):
#         actor = Sequential()
        input_x = (Input(self.state_size))
        x = Flatten(input_shape=self.state_size)(input_x)
        x = Dense(512, activation='elu', kernel_initializer='he_uniform')(x)
        
        action = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(x)
        value = Dense(1, activation='linear', kernel_initializer='he_uniform')(x)

        def ppo_loss(y_true, y_pred):
            # Defined in https://arxiv.org/abs/1707.06347
            advantages, prediction_picks, actions = y_true[:,:1], y_true[:,1:1+self.action_size], y_true[:,1+self.action_size:]
            LOSS_CLIPPING = 0.2
            ENTROPY_LOSS = 5e-3

            prob = y_pred * actions
            old_prob = actions * prediction_picks
            r = prob / (old_prob + 1e-10)
            p1 = r * advantages
            p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1+LOSS_CLIPPING) * advantages
            loss = -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS* -(prob * K.log(prob + 1e-10)))
            
            return loss

        actor = Model(inputs=input_x, outputs=action)
        critic = Model(inputs=input_x, outputs=value)

        actor.compile(loss=ppo_loss, optimizer=RMSprop(lr=self.lr))
        critic.compile(loss='mse', optimizer=RMSprop(lr=self.lr))

        return actor, critic

    def calculate_average(self, score):
        self.scores.append(score)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

        return self.average[-1]
    
    def save(self):
        self.actor.save(self.actor_name + '.h5')
        self.critic.save(self.critic_name + '.h5')
    
    def load(self):
        self.actor.load_weights(self.actor_name + '.h5')
        self.critic.load_weights(self.critic_name + '.h5')

    def check_or_create_save_path(self):
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)

    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for x in range(STACK_SIZE):
            state = self.stack_frames(frame, image_memory)

        return state
    
    def step(self, action, env, image_memory):
        next_state, reward, done, _ = env.step(action)
        next_state = self.stack_frames(next_state, image_memory)

        return next_state, reward, done
    
    def stack_frames(self, frame, image_memory):
        if image_memory.shape == (1, *self.state_size):
            image_memory = np.squeeze(image_memory)

        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != 80 or frame_cropped.shape[1] != 80:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255
        # converting to RGB (OpenCV way)
        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        image_memory = np.roll(image_memory, 1, axis = 0)

        # inserting new frame to free space
        image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        return np.expand_dims(image_memory, axis=0)

    def discount_and_normalize_reward(self, reward):
        # apply the discount and normalize it to avoid big variability os rewards
        discounted_reward = np.zeros_like(reward)
        cummulative = 0.0

        for i in reversed(range(len(reward))):
            if reward[i] != 0:
                cummulative = 0.0       # reset the summation
            cummulative = cummulative * self.gamma + reward[i]
            discounted_reward[i] = cummulative

        mean = np.mean(discounted_reward)
        std = np.std(discounted_reward)
        discounted_reward = (discounted_reward - mean)/std
        
        return discounted_reward
    
    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        prediction = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)

        return action, prediction

    pylab.figure(figsize=(18,9))
    def plot_model(self, score, episode):
        if str(episode)[-2:] == '00': # much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Scores', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path + '.png')
            except OSError:
                pass

    def replay(self, states, actions, rewards, predictions):
        #reshape memory to appropriate shape for trainning
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        discounted_r = np.vstack(self.discount_and_normalize_reward(rewards))
        
        # get critic network predictions
        values = self.critic.predict(states)
        # comput advantages
        advantages = discounted_r - values

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])

        # training actor and critic networks
        self.actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=len(rewards))
        self.critic.fit(states, discounted_r, epochs=self.epochs, verbose=0, shuffle=True, batch_size=len(rewards))

    def run(self):
        for episode in range(MAX_EPISODES):
            state = self.reset(self.env)
            score, saving = 0, ''

            # Instantiate games memory
            states, actions, rewards, predictions = [], [], [], []
            while True:
                # self.env.render()
                # Actor picks an action
                action, prediction = self.act(state)
                next_state, reward, done = self.step(action, self.env, state)
                
                # self.memorize(stacked_state, action, reward)
                states.append(state)
                action_p = np.zeros([self.action_size])
                action_p[action] = 1
                actions.append(action_p)
                rewards.append(reward)
                predictions.append(prediction)

                #update current state
                state = next_state
                score += reward

                if done:
                    average = self.calculate_average(score)
                    self.plot_model(score, episode)
                    if average >= self.max_average:
                        saving = 'SAVING'
                        self.max_average = average
                        self.save()
                    else:
                        saving = ''

                    print('Episode: {}/{} Score: {} Average: {:.2f} {}'
                    .format(episode, MAX_EPISODES, score, average, saving))
    
                    self.replay(states, actions, rewards, predictions)    
                    break

        self.env.close()

    
    def train(self, n_threads):
        self.env.close()
        # instatiate one environment per thread
        envs = [gym.make(self.env_name) for _ in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_treading,
                daemon=True,
                args=(self, envs[i], 
                i)) for i in range(n_threads)]
        
        for t in threads:
            time.sleep(2)
            t.start()
        
        try: 
            for t in threads:
                time.sleep(4)
                t.join()
                
        except (KeyboardInterrupt, SystemExit):
                # Daemon seem doesn't work with kaggle, so it's necessary switch a global flag to 
                # finish the threads using a safe mode. In order hand, all resources allocated
                # by threads will keept used and maybe the own threads don't finish them.
                # That's ocorred when I used kaggle, when I tried finish just the main thread the 
                # other threads keept runing even using True Daemon. So mt solution was use a flag
                print("########### Exiting all threads...It may take a while ###########")
                self.keep_running_thread = False
                for t in threads:
                    t.join()
                print('All threads are finished....')
                
                # Release resources allocated during training
                self.session.close()
                
                # Release GPU memory
                device = cuda.get_current_device()
                device.reset()
            

    def train_treading(self, agent, env, thread):
        try:
            with self.session.as_default():
                while self.episode < MAX_EPISODES:
                    # Reset episodes
                    score, saving = 0, ''
                    state = self.reset(env)

                    # Instantiate or reset games memory
                    states, actions, rewards, predictions = [], [], [], []
                    while True:
                        # The raise inside the threads turn possible break a lot of while at once,
                        # and the try/except enable doesn't generate a exceptoion error message
                        if not self.keep_running_thread: raise KeyboardInterrupt
                        action, prediction = agent.act(state)
                        next_state, reward, done = self.step(action, env, state)

                        states.append(state)
                        action_p = np.zeros([self.action_size])
                        action_p[action] = 1
                        actions.append(action_p)
                        rewards.append(reward)
                        predictions.append(prediction)

                        score += reward
                        state = next_state

                        if done:
                            break

                    self.lock.acquire()
                    self.replay(states, actions, rewards, predictions)
                    self.lock.release()

                    # Update episode count
                    with self.lock:
                        average = self.calculate_average(score)
                        self.plot_model(score, self.episode)
                        # saving best models
                        if average >= self.max_average:
                            self.max_average = average
                            self.save()
                            saving = 'SAVING'

                        else:
                            saving = ''
                        print('Episode: {}/{}, Thread: {}, Score: {}, Average: {:.2f} {}'
                        .format(self.episode, MAX_EPISODES, thread, score, average, saving))

                        if (self.episode < MAX_EPISODES):
                            self.episode += 1
                env.close()
        except KeyboardInterrupt:
            print('Thread {} finished.'.format(thread))

    def test(self, actor_name, critic_name):
        self.load()
        for e in range(100):
            state = self.reset(self.env)
            score = 0

            while True:
                action = np.argmax(self.actor.predict(state))
                state, reward, done = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
            
        self.env.close()

# %% [code]
# That's a good way to automate the choice of number of workers
import multiprocessing
n_workers = multiprocessing.cpu_count()
print(n_workers)

# %% [code]
env_name = 'Pong-v0' 
agent = PGAgent(env_name)
agent.train(n_threads = n_workers)
# agent.test()