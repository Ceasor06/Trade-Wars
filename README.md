# Trade-Wars


![image](https://github.com/Ceasor06/Trade-Wars/assets/105945382/e56d9574-a20b-4311-a740-e8216114f34d)

**Investigating the Application of Multi-Agent Reinforcement Learning Algorithms in a Custom Stock Trading Environment Using Historical Data from S&P 100 Index.**

## This project was part of my coursework: Reinforcement Learning under the guidance of Prof. Alina Vereshchaka @ University at Buffalo.
<img align="right" width=200 src="https://github.com/Ceasor06/Trade-Wars/blob/main/7cb45faf3f3893fcb6590466ef69a51a.jpg" />

The "RL Trade Wars: Investments Across Timeframes" project explores the application of various reinforcement learning (RL) algorithms to algorithmic trading strategies in a competitive market environment. This project aims to evaluate how different RL-based trading algorithms perform and adapt when faced with dynamic market conditions and interactions among multiple agents. By simulating a stock trading environment with historical data from the S&P 100 index, we seek to uncover insights into the effectiveness and adaptability of RL methods in making investment decisions.

The project involves the development of a custom trading environment where multiple RL agents, each employing distinct algorithms, interact with the market. We implemented and tested several RL algorithms, including Deep Q-Learning (DQN), Double Deep Q-Learning (DDQN), and Advantage Actor-Critic (A2C), to understand their behavior and performance in this simulated trading scenario.

Our findings reveal the strengths and limitations of each algorithm, providing valuable insights into their suitability for real-world trading applications. The ultimate goal is to identify strategies that can potentially lead to more profitable and stable trading outcomes in a competitive financial market.

<hr>

## Gym - Environment

Our custom trading environment, named 'StockEnvironment', is designed following the gymnasium standards to simulate a realistic stock trading scenario. In this environment, multiple RL agents interact, each utilizing a different algorithm to make trading decisions. 

The primary components of the environment include initialization, action space, observation space, and the methods to reset and step through the environment.

<br>
</br>


<img width="569" alt="image" src="https://github.com/Ceasor06/Trade-Wars/blob/main/39b25e27-04c6-40b3-8fbd-4e48b06b3253.png">


<img width="569" alt="image" src="https://github.com/Ceasor06/Trade-Wars/blob/main/2a23f4dc-16b9-4a8c-b21f-57b4eadad235.png">

<hr>

## Models used

- Double - DQN : [Model1](https://github.com/Ceasor06/Trade-Wars/blob/main/Algorithms/DDQN_1.ipynb)

- DQN : [Model2](https://github.com/Ceasor06/Trade-Wars/blob/main/Algorithms/DQN_1.ipynb)

- A2C : [Model3](https://github.com/Ceasor06/Trade-Wars/blob/main/Algorithms/A2C_25.ipynb)

<hr>

## Future Work



- Using real time data using satellites to make accurate and localised predictions.

