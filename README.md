# compsci-685-project-semantle
This Github page houses all of the relevant files for the research project for Dr. Chang's COMPSCI 685 [Advanced NLP] 
course for Spring 2025. This project attempts to apply reinforcement learning to create an AI designed to solve
games of Semantle.

## Instructions
* Downloading the GoogleNews word2vec model is required. It is used by Semantle to evaluate similarity between words.
 You can find the download for the model [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g).
* An OpenAI API Key is required to run the LLM Baseline.
  * Generate a new API key at https://platform.openai.com, then store it in a `.env` file with the variable name OPENAI_API_KEY.

## File Directory
* `semantle_logs/` - Houses JSON files for results from 200 test games for LLM Baseline.
* `Semantle-LLM-Baseline.ipynb` - Python notebook that houses code to both extract words from past Semantles, as well as running the GPT-4o baseline.
* `Semantle-RL.py` - Original iteration of the reinforcement learning algorithm as a Python executable.
* `Semantle-RL2.ipynb` - Final RL algorithm
* `Semantle-notebook.ipynb` - Extracting data for RL algorithms.
* `Semantle_Gameplay.ipynb` - Initial gameplay algorithm for the RL alg.
* `count_1w.txt` - List of most common unigrams, according to [Google ngrams analysis done by Peter Norvig](https://norvig.com/ngrams/).
* `mdp.py` - Initial algorithm for a Markov Decision Process, for RL algorithm (obsolete).

## Credits
* Frankie Furnelli (ffurnelli [at] umass [dot] edu)
* Eric Nunes (eenunes [at] umass [dot] edu)
* Thomas Potts (tpotts [at] umass [dot] edu)
* Virtulya Rajput (vrajput [at] umass [dot] edu)