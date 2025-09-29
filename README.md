# 🚗 Driving AI with Deep Q-learning

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.7.1%2Bcu118-red.svg)
![Numpy](https://img.shields.io/badge/Numpy-2.2.6-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-red.svg)
![OpenPyxl](https://img.shields.io/badge/OpenPyxl-3.1.5-red.svg)  

![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)  

## 📝 Project Description 
This project showcases an AI that learns to drive a car in a 2D environment using the deep Q-learning algorithm. No hardcoded pathfinding — the agent improves by trial, error, and reward-based learning. 🧠📈


---

## 🚀 Features
  🤖 Uses Deep Q-Learning with experience replay and epsilon-greedy exploration

  🧱 Neural network approximates Q-values for discrete actions (e.g., accelerate, turn left/right)


## Example Outputs
Here is an image of what it looks like :

![Image_cars](Images/Img_car.png)


### 📝 Notes & Observations
⏳ Training is unstable at first — the car often spins out or crashes quickly — but over time, it learns to stabilize, turn properly, and sometimes follow simple roads or avoid walls.

🎛️ **Hyperparameters** (learning rate, epsilon decay, reward shaping) have a **huge impact** on learning performance.

Here, we can see that over 100 steps, the best path have been found (in just more than 5 min).

It is more **hesitant** for the borrowed path but **adapts better** to different circuits than **Genetic algorithm such as [AI_driving_genetic_version](https://github.com/Thibault-GAREL/AI_driving_genetic_version)

---

## ⚙️ How it works
🎮 The AI controls a car in a Pygame environment with basic physics and obstacles. The game is my [driving_game](https://github.com/Thibault-GAREL/driving_game) !

🧠 It uses a Deep Q-Network to estimate the best action to take from any given state.

🧾 Inputs include distance to next checkpoint, velocity, distance to obstacles, and relative angles to next checkpoint.

🎯 Rewards are given based on life time, distance to the next checkpoint, avoiding collisions, velocity to encourage speed and reaching checkpoints.






