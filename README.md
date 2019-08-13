# Flappy-bird-agent
This project is dedicated teaching a Reinforcement learning agent to play Flappy Bird game using DQN and PyGame Learning Environment.

![gameplay](gifs/ATARI_192388_reward_219.0.gif)

## Project setup:
### 1. Install PLE:
```
$ git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
$ cd PyGame-Learning-Environment/
$ pip install -e .
```
### 2. Install PyGame following the instructions [here](http://www.pygame.org/wiki/GettingStarted#Pygame%20Installation)
### 3. Create your virtual environment 
```
$ cd setup/
$ conda env create -f environment.yml
```

### To train the model activate your virtual environment and run `train.py` script:

```
python train.py
```

Script parameters: 

`--batch_size`, `-b` - size of a minibatch, default is 1024

`--gamma`, `-g` - discount factor, default is 0.99

`--model`, `-m` - Model type to train (supported types: `dqn` and `q_learn`)

### To test already trained model activate your virtual environment and run `evaluate.py` script: 

```
python evaluate.py
```

Script parameters: 

`--model`, `-m` - Model type to evaluate.

`--model_path`, `-path` - path to a saved model

`--gif_path` - path to a folder, where to store the gif output

`--num_episodes`, `-n` - number of evaluation episodes (default = 100)

