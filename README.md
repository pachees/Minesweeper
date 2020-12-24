# Minesweeper Generator and Solver

There are two major components to this project, drawing from its name:

* **Real time grid generator**: Interface to spontaeously generate masked Minesweeper grids of any size and any specification for the training of an AI model
  * This is helpful when trying to train predictive models to solve Minesweeper games
  * Most important benefit is that generation happens in **real time**, meaning the memory overhead of training data remains **virtually zero**
* **TF-1 pipeline**: Rudimentary pipeline written with Tensorflow v1 that implements real time neural net training 
  * They implement simple feedforward MLPs written with an intent of building TF structures from first principles 
  * Better algorithms can always be used for this (current method dates far enough back to use tf contrib), especially reinforcement learning approaches
    * Please raise a PR if you want to take this forward
  
The project wasn't designed for public use so it is not modelled as a package, but the interface on each file is very flexible. Each parameter can be tuned to suit how you want to train your model. 

