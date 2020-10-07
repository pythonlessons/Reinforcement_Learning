# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
import threading
from threading import Thread, Lock
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic

class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = 20000, 0, -21.0 # specific for pong
        self.lock = Lock()
        self.lr = 0.000025

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self, states, actions, rewards):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(rewards)

        # Get Critic network predictions
        value = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - value
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
 
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path+".png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.Model_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self, frame, image_memory):
        if image_memory.shape == (1,*self.state_size):
            image_memory = np.squeeze(image_memory)
            
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
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
        #self.imshow(image_memory,0)
        #self.imshow(image_memory,1)
        #self.imshow(image_memory,2)
        #self.imshow(image_memory,3)
        
        return np.expand_dims(image_memory, axis=0)

    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage(frame, image_memory)
        return state

    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        next_state = self.GetImage(next_state, image_memory)
        return next_state, reward, done, info
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.reset(self.env)
            done, score, SAVING = False, 0, ''
            # Instantiate or reset games memory
            states, actions, rewards = [], [], []
            while not done:
                #self.env.render()
                # Actor picks an action
                action = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action, self.env, state)
                # Memorize (state, action, reward) for training
                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))

                    self.replay(states, actions, rewards)
         # close environemnt when finish training  
        self.env.close()

    def train(self, n_threads):
        self.env.close()
        # Instantiate one environment per thread
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self,
                    envs[i],
                    i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()
            
        for t in threads:
            time.sleep(10)
            t.join()
            
    def train_threading(self, agent, env, thread):
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.reset(env)
            # Instantiate or reset games memory
            states, actions, rewards = [], [], []
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = self.step(action, env, state)

                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                
                score += reward
                state = next_state

            self.lock.acquire()
            self.replay(states, actions, rewards)
            self.lock.release()
                    
            # Update episode count
            with self.lock:
                average = self.PlotModel(score, self.episode)
                # saving best models
                if average >= self.max_average:
                    self.max_average = average
                    self.save()
                    SAVING = "SAVING"
                else:
                    SAVING = ""
                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, thread, score, average, SAVING))
                if(self.episode < self.EPISODES):
                    self.episode += 1
        env.close()            

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        self.env.close()

if __name__ == "__main__":
    env_name = 'PongDeterministic-v4'
    #env_name = 'Pong-v0'
    agent = A3CAgent(env_name)
    #agent.run() # use as A2C
    agent.train(n_threads=5) # use as A3C
    #agent.test('Models/Pong-v0_A3C_2.5e-05_Actor.h5', '')
