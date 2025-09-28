# 🚗 Driving AI with Deep Q-learning
This project showcases an AI that learns to drive a car in a 2D environment using the deep Q-learning algorithm. No hardcoded pathfinding — the agent improves by trial, error, and reward-based learning. 🧠📈

  
# 🧠 What It Does
🎮 The AI controls a car in a Pygame environment with basic physics and obstacles.

🧠 It uses a Deep Q-Network to estimate the best action to take from any given state.

🧾 Inputs include distance to next checkpoint, velocity, distance to obstacles, and relative angles to next checkpoint.

🎯 Rewards are given based on life time, distance to the next checkpoint, avoiding collisions, velocity to encourage speed and reaching checkpoints.

  
# 🚀 Features
  🤖 Uses Deep Q-Learning with experience replay and epsilon-greedy exploration

  🧱 Neural network approximates Q-values for discrete actions (e.g., accelerate, turn left/right)


Here is an image of what it looks like :

![Image_cars](Images/Img_car.png)

# 📦 Dependencies
  - Python 3.x 🐍
  - pytorch for neurons 🧠
  - numpy for tracking and plotting results 📊
  - pygame for visualization 🎮

# 📝 Notes & Observations
⏳ Training is unstable at first — the car often spins out or crashes quickly — but over time, it learns to stabilize, turn properly, and sometimes follow simple roads or avoid walls.

🎛️ Hyperparameters (learning rate, epsilon decay, reward shaping) have a huge impact on learning performance.

Here, we can see that over 100 steps, the best path have been found (in just more than 5 min).


