# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:31:39 2023

@author: Sankhapani
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:09:36 2023

@author: Sankhapani
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, learning_rate=0.01, hidden_size=3):
        # Neural network parameters
        self.input_size = 1
        self.hidden_size = hidden_size
        self.output_size = 1
        self.learning_rate = learning_rate

        # Neural network weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

        # Data for visualization
        self.predicted_output = 0.5
        self.original_data = []
        self.predicted_data = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # Hidden layer with relu activation
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)

        # Output layer with sigmoid activation
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)

        return self.predicted_output

    def backward_propagation(self, inputs, targets):
        # Output layer
        output_error = targets - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        # Hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, inputs, targets, epochs=1000):
        for epoch in range(epochs):
            # Forward propagation
            self.forward_propagation(inputs)

            # Backward propagation and optimization
            self.backward_propagation(inputs, targets)

    def forecast_health(self, future_data):
        inputs = np.array([[future_data]])
        # Forward propagation
        self.predicted_output = self.forward_propagation(inputs)

        return self.predicted_output

def generate_data(df, threshold=0.7):
    values = df["Sales"].values
    labels = ["Healthy" if value < threshold else "Unhealthy" for value in values]

    return pd.DataFrame({"Sales": values, "Label": labels})

def main():
    st.title("Asset Health Forecasting Neural Network")
    st.write("Predict asset health from time-series data") 
    st.write("Please use this csv https://docs.google.com/spreadsheets/d/1USQGb9J_xTNIIndzf3REBBg5yT60ApabmG0MD_LLfkI/edit?usp=sharing")
    st.write("GROUP 7: KAUSHIK DAS ,SANKHAPANI NEOG, RISHAB BORA, PRIYANUJ BORAH, DIGBIJOY CHETRY") 
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.line_chart(data["Sales"], use_container_width=True)

        learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        hidden_size = st.sidebar.slider("Hidden Layer Size", min_value=1, max_value=10, value=3, step=1)

        neural_network = NeuralNetwork(learning_rate=learning_rate, hidden_size=hidden_size)

        neural_network.original_data = data["Sales"].tolist()

        inputs = np.array(data["Sales"][:-1]).reshape(-1, 1)
        targets = np.array(data["Sales"][1:]).reshape(-1, 1)
        neural_network.train(inputs, targets)

        threshold = 0.5

        if st.button("Predict"):
            for _ in range(10):
                prediction = neural_network.forecast_health(neural_network.original_data[-1])
                neural_network.original_data.append(prediction[0])
                neural_network.predicted_data.append(prediction[0])

            labels = ["Healthy" if pred < threshold else "Unhealthy" for pred in neural_network.predicted_data]

            st.line_chart(list(neural_network.predicted_data), use_container_width=True)
            st.write("Labels:", labels)

            fig, ax = plt.subplots()
            ax.plot(neural_network.original_data, label='Original Data', color='blue')
            ax.plot(range(len(data), len(data) + 10), neural_network.predicted_data, label='Predicted Data', color='orange')
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Sales')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
