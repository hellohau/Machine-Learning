class MLP {
    constructor(inputSize, hiddenSize, outputSize) {
      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      this.outputSize = outputSize;
  
      // Initialize weights and biases
      this.weights1 = this.randomMatrix(inputSize, hiddenSize, -1, 1);
      this.biases1 = this.zeroMatrix(1, hiddenSize);
      this.weights2 = this.randomMatrix(hiddenSize, outputSize, -1, 1);
      this.biases2 = this.zeroMatrix(1, outputSize);
    }
  
    // Sigmoid activation function
    sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    }
  
    // Mean squared error loss function
    mse(output, target) {
      let error = 0;
      for (let i = 0; i < output.length; i++) {
        error += Math.pow(output[i] - target[i], 2);
      }
      return error / output.length;
    }
  
    // Forward pass through the network
    forward(input) {
      this.hidden = [];
      for (let j = 0; j < this.hiddenSize; j++) {
        let sum = 0;
        for (let i = 0; i < this.inputSize; i++) {
          sum += input[i] * this.weights1[i][j];
        }
        sum += this.biases1[0][j];
        this.hidden.push(this.sigmoid(sum));
      }
  
      this.output = [];
      for (let j = 0; j < this.outputSize; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenSize; i++) {
          sum += this.hidden[i] * this.weights2[i][j];
        }
        sum += this.biases2[0][j];
        this.output.push(this.sigmoid(sum));
      }
  
      return this.output;
    }
  
    // Backward pass through the network
    backward(input, target, learningRate) {
      // Calculate error and delta for output layer
      const outputError = [];
      const outputDelta = [];
      for (let j = 0; j < this.outputSize; j++) {
        outputError.push(target[j] - this.output[j]);
        outputDelta.push(outputError[j] * this.output[j] * (1 - this.output[j]));
      }
  
      // Calculate error and delta for hidden layer
      const hiddenError = [];
      const hiddenDelta = [];
      for (let j = 0; j < this.hiddenSize; j++) {
        let sum = 0;
        for (let i = 0; i < this.outputSize; i++) {
          sum += outputDelta[i] * this.weights2[j][i];
        }
        hiddenError.push(sum);
        hiddenDelta.push(hiddenError[j] * this.hidden[j] * (1 - this.hidden[j]));
      }
  
      // Update weights and biases
      for (let j = 0; j < this.outputSize; j++) {
        for (let i = 0; i < this.hiddenSize; i++) {
          this.weights2[i][j] += learningRate * outputDelta[j] * this.hidden[i];
        }
        this.biases2[0][j] += learningRate * outputDelta[j];
    }

        for (let j = 0; j < this.hiddenSize; j++) {
        for (let i = 0; i < this.inputSize; i++) {
            this.weights1[i][j] += learningRate * hiddenDelta[j] * input[i];
        }
        this.biases1[0][j] += learningRate * hiddenDelta[j];
        }

    }

    // Helper function to generate a matrix with random values between min and max
    randomMatrix(rows, cols, min, max) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix.push([]);
            for (let j = 0; j < cols; j++) {
                matrix[i].push(Math.random() * (max - min) + min);
            }
        }

        return matrix;
    }

    // Helper function to generate a matrix filled with zeros
    zeroMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix.push([]);
            for (let j = 0; j < cols; j++) {
                matrix[i].push(0);
            }
        }

        return matrix;
    }
}