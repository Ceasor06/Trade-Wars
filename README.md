# Trade-Wars


![image](https://github.com/Ceasor06/Trade-Wars/assets/105945382/e56d9574-a20b-4311-a740-e8216114f34d)

**Investigating the Application of Multi-Agent Reinforcement Learning Algorithms in a Custom Stock Trading Environment Using Historical Data from S&P 100 Index.**

## This project was part of my coursework: Reinforcement Learning under the guidance of Prof. Alina Vereshchaka @ University at Buffalo.
<img align="right" width=200 src="https://github.com/Ceasor06/Trade-Wars/blob/main/7cb45faf3f3893fcb6590466ef69a51a.jpg" />

Floods are a recurrent phenomenon, the most common natural disaster in India, which cause huge loss of lives and 
damage to livelihood systems, property, 
infrastructure and public utilities. 
On average nearly 1600 people die every year due to floods, 75 lakh hectares of land is affected and the damage caused to crops, 
houses and public utilities is Rs. 1805 crores due to floods.
Flood prediction is difficult to predict due to its nonlinear and dynamic nature, 
various researchers have tackled this problem using different techniques ranging from physical models to image processing, 
we picked up time series forecasting by creating our own dataset from scratch extracted from different stations and government of India websites.

<hr>

## Dataset Used 

- We took data from the [Indian Flood Inventory](https://link.springer.com/article/10.1007/s11069-021-04698-6) to find the dates for floods.

- We extracted the information of Rain statewise data from [India Environment Portal](http://www.indiaenvironmentportal.org.in/media/iep/infographics/Rainfall%20in%20India/112%20years%20of%20rainfall.html).

- We have also extracted information of population from [world bank](https://data.worldbank.org/indicator/SP.POP.TOTL?locations=IN)  

- and [temperature](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data ) and made a dataset using all these features to perform our time series analysis.

<br>
</br>

_Attributes of our Dataset:_

<img width="569" alt="image" src="https://user-images.githubusercontent.com/105945382/211894966-895f0fe1-d009-46c2-ba97-2e4282ab525c.png">



<hr>

## Models used

- XG-B Classifier : [Model1](https://github.com/Ceasor06/Machine-Learning-aided-Flood-Forecasting/tree/main/Model1)

- Random Forest : [Model2](https://github.com/Ceasor06/Machine-Learning-aided-Flood-Forecasting/tree/main/Model2)

- Artificial Neural Network : [Model3](https://github.com/Ceasor06/Machine-Learning-aided-Flood-Forecasting/tree/main/Model3)

<hr>

## Future Work

- There is still need of more quality data flood of India, especially of rural areas where impact of flood is more devastating than other areas. 

- Creating a proper dataset having more attributes will make our model more accurate than before which in turn will help our client achieve accurate results.

- Using real time data using satellites to make accurate and localised predictions.

