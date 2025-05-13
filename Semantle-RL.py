import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from nltk import FreqDist, download
from nltk.corpus import brown
import nltk
download('brown')
frequency_list = FreqDist(word for word in brown.words() if word.islower())
word_list = [word for word,_ in frequency_list.most_common(20000)]
# nltk.download('words')
# from nltk.corpus import words
# word_list = words.words()
""" This class is a Gymnasium Environment for Semantle, similar to Thomas' MDP """
class SemantleEnv(gym.Env):
    def __init__(self, target_words, embedding_model=None, target_word=None):
        super(SemantleEnv, self).__init__()
        self.target_words = target_words
        if embedding_model == None:
            embedding_model = api.load('word2vec-google-news-300')
        self.embedding_model = embedding_model  # e.g., a Word2Vec or BERT wrapper
        self.word_list = [word for word in word_list if word in self.embedding_model]
        if target_word == None or target_word not in self.target_words:
            target_word = random.sample(self.target_words, k=1)[0]
        self.target_word = target_word
        self.target_vector = self.get_embedding(target_word)

        self.action_space = spaces.Discrete(len(self.word_list))  # index of word list
        #self.observation_space = spaces.Box(low=-1.,high=1.,shape=(300,))
        # self.observation_space = spaces.Dict({'current_embedding': spaces.Box(low=-1., high=1., shape=(300,), dtype=np.float32), 
        #                                       'score': spaces.Box(low=-1., high=1., dtype=np.float32),
        #                                       'best_embedding': spaces.Box(low=-1., high=1., shape=(300,), dtype=np.float32),
        #                                       'max_score': spaces.Box(low=-1., high=1., dtype=np.float32),
        #                                       'delta_score': spaces.Box(low=-1., high=1., dtype=np.float32),
        #                                       'number_of_guesses': spaces.Box(low=0, high=300)})
        self.observation_space = spaces.Dict({'best_embedding': spaces.Box(low=-1., high=1., shape=(300,), dtype=np.float32),
                                              'max_score': spaces.Box(low=-1., high=1., dtype=np.float32),
                                              'number_of_guesses': spaces.Box(low=0, high=300)})

        self.state = []
        self.guess_history = []
        self.best = None

    def get_embedding(self, word):
        return self.embedding_model[word]
    
    def sample_target_word(self):
        return random.sample(self.target_words, k=1)[0]

    def compute_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def reset(self, seed=None, options=None):
        self.guess_history = []
        self.target_word = self.sample_target_word()
        self.target_vector = self.get_embedding(self.target_word)
        print(self.target_word)
        self.state = {#'current_embedding': np.zeros((300,)), 
        #                 'score': np.array([-1]),
                         'best_embedding': np.zeros((300,)),
                         'max_score': np.array([-1]),
        #                 'delta_score': np.array([0]),
                         'number_of_guesses': np.array([0])}
        
        #self.state = np.zeros(self.observation_space.shape)
        self.best = None
        return self.state, {}

    def step(self, action):
        word = self.word_list[action]
        guess_vector = self.get_embedding(word)
        similarity = self.embedding_model.similarity(word, self.target_word)#self.compute_similarity(guess_vector, self.target_vector)
        print(f'guess: {word}, sim: {similarity}')
        self.guess_history.append((word, similarity))
        if self.best == None or similarity > self.best[1]:
            self.best = (word,similarity)
        delta = similarity - self.best[1] if self.best != None else 0
        #delta = (self.guess_history[-1][1]-self.guess_history[-2][1]) if len(self.guess_history) >= 2 else 0
        #self.state = self.get_embedding(self.best[0])
        self.state = {#'current_embedding': guess_vector, 
                        #'score': np.array([similarity], dtype=float),
                        'best_embedding': self.get_embedding(self.best[0]),
                        'max_score': np.array([self.best[1]], dtype=float),
                        #'delta_score': np.array([delta], dtype=float),
                        'number_of_guesses': np.array([len(self.guess_history)], dtype=float)}

        ## Reward Modeling
        reward = (similarity)*100 #+ delta  # or reward shaping if desired
        reward *= 2 if self.embedding_model.similarity(self.target_word, self.best[0]) >= 0.7 else 1
        print(reward)
        terminated = similarity >= 0.99  # or similarity == 1.0
        truncated = len(self.guess_history) == 150

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.guess_history:
            print(f"Guesses: {self.guess_history[-5:]}")

    def close(self):
        pass

""" The following code tries to learn a PPO model """
from stable_baselines3 import PPO
import pandas
df = pandas.read_csv("semantle.csv") # Pull in target words
env = SemantleEnv(target_words=df['answer'].to_list())

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=25_000)
obs, info = env.reset()
print(f"Target: {env.target_word}")
for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    #env.render()
    # VecEnv resets automatically
    if done or truncated:
        obs, info = env.reset()
        print(f"Target: {env.target_word}")

env.close()