# Deep Learning Book Club
This repo will contain a few exercises and programs for the study of GoodFellow, Bengio and Courville's Deep Learning book.

Follow us on youtube at youtube.com/c/MeetScience
Follow us on telegram at t.me/Scienza

## LearningXOR.py

From section 6.1, a neural network is implemented to learn the function XOR, a classical non linear problem in machine learning.

We will use a single hidden layer.


### Requisiti

- Python 3.6+
- NumPy

### Installazione

1. Clone this repo:
   ```
   git clone https://github.com/mechanapoleon/DeepLearningBookClubExamples.git
   ```

2. (Optional) Create a virtual environment:
   - Windows:
     ```
     python -m venv venv
     .\venv\Scripts\Activate
     ```
   - macOS/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### How to Run

To run the neural network:

```
python LearningXOR.py
```

This script will train the neural network and show the final output.

### Project structure

- `LearningXOR.py`: Chapter 6.1 neural net
- `requirements.txt`: Dependencies.
- `README.md`: This file.

### How it works

1. The XOR dataset is provided as an input.
2. The NN is initialised with random weights.
3. Traing: the NN uses forward propagation, compute the error using the Mean Squared Error (MSE), and then uses backpropagation to update the weights.
4. After training, the NN should be able to approximate XOR correctly.

## Experiment

Play with  `LearningXOR.py` and modify:

- `learning_rate`
- `epochs`: number of training iterations
- the NN structure (what will happen if you change the number of hidden layer neurons)