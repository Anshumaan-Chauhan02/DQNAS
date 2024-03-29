<h2 align='center'><b> DQNAS: Neural Architecture Search using Reinforcement Learning </b></h2>
<img width='1500' height='300' src="https://user-images.githubusercontent.com/98472023/216144494-88d8b907-0c01-4956-9aee-a0de4dc6da65.png" alt="my banner"> 

<h4 align='center'> Project Description </h4> 
Convolutional Neural Networks have been used in a variety of image related applications after their rise in popularity due to ImageNet competition. Convolutional Neural Networks have shown remarkable results in applications including face recognition, moving target detection and tracking, classification of food based on the calorie content and many more. Designing of Convolutional Neural Networks requires experts having a cross domain knowledge and it is laborious, which requires a lot of time for testing different values for different hyperparameter along with the consideration of different configurations of existing architectures. Neural Architecture Search is an automated way of generating Neural Network architectures which saves researchers from all the brute-force testing trouble, but with the drawback of consuming a lot of computational resources for a prolonged period. In this paper, we propose an automated Neural Architecture Search framework DQNAS, guided by the principles of Reinforcement Learning along with One-shot Training which aims to generate neural network architectures that show superior performance and have minimum scalability problem. 
<br>

<h4> Paper link : https://arxiv.org/abs/2301.06687</h4>
<br>
This Repository consist of code and documentation needed for successfully running the project. <br>
Below are the steps needed to be installed before running this project : 

### Technical Skills 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
<br>
## 1) Installing Anaconda
    https://docs.anaconda.com/anaconda/install/index.html

## 2) Installing Spyder
#### Check the following installation guide
    https://docs.spyder-ide.org/current/installation.html
    
## 3) Installing Machine Learning Libraries
##### TensorFlow
      !pip install tensorflow
##### Keras
      !pip install keras
##### Pandas
      !pip install pandas
##### NumPy
      !pip install numpy
##### Matplotlib
      !pip install matplotlib
     
## 4) File Description and Content 
* CNNCONSTANTS.py : Contains values for constants used in CNN and LSTM model (Controller) generation anc compilation
* CNNGenerator.py : Functions used for the creation of CNN architecture 
* DQNAgent.py : Consists functions required for generation of controller model
* DQNController.py : Sampling of CNN architectures and training of the controller model 
* NASrun.py : Main file used to run the program (May change the dataset loaded from Keras to some other dataset)
* NASutils.py : Printing and Visualization of CNN architectures 
* cnnas.py : Training and Testing of CNN Architectures 

<br><br><br>
![Anurag’s github stats](https://github-readme-stats.vercel.app/api?username=Anshumaan-Chauhan02)
![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=Anshumaan-Chauhan02&layout=compact)
