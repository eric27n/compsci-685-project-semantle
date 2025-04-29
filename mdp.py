import numpy as np
import random
# Semantic model:
import gensim.downloader as api
# from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

class SemantleMDP:
    def __init__(self, vec_model, goal, epsilon, max_iters):
        ## Are we using the gensim KeyedVectors Word2Vec model?? 
        self.vec_model = vec_model # currently assuming its a KeyedVectors model
            # ^ Expecting vec_model = api.load('word2vec-google-news-300')
        self.goal  = goal # goal word
        self.epsilon = epsilon # for learning / exploration?
        self.max_iters = max_iters # cap for number of guesses

        self.state = None # s = 'word', might change to vector embedding('word') for Neural Net?
        self.reward = None # semantle score given state
        self.iters = 0 
        self.vocab = list(vec_model.key_to_index.keys()) # word list!

    def reset(self):
        # Random initial state
        guess = random.choice(self.vec_model.vocab) #check if this works
        self.state = guess
        self.iters = 0
        return guess # returns s0
    
    def step(self, guess):
        self.state = guess
        self.iters += 1

        # Similarity func based on KeyedVectors, change if we're using smthing diff 
        reward = self.vec_model.similarity(guess, self.goal)
        self.reward = reward

        is_done = (guess == self.goal or self.iters >= self.max_iters) # if True, end episode
        return guess, reward, is_done

    def guess(self, rl_model):
        # need information about our RL model!!
        return 