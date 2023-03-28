# Neuroevolution Wordle Bot (NEWB)

This is a simple project to demonstrate the use of neuroevolution to create a bot that can play the game Wordle. The bot is trained using a genetic algorithm to evolve a neural network using the [NEAT-Python](https://neat-python.readthedocs.io/en/latest/) library.

## setup

I've been using [Anaconda](https://www.anaconda.com/) to manage my python environments. I've included a `environment.yml` file that can be used to create a conda environment with all the required dependencies. To create the environment run the following command:

```bash
conda env create -f environment.yml
```

# scripts

## training

running training

```bash
python train.py
```

## playing

using the bot interactively

```bash
python play.py
```

## show

show stats on current training

```bash
python show.py
```