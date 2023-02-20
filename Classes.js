const math = require('mathjs');

class Perceptron{
  
    w1 = 0.5;
    w2 = 0.5;
    b = 1;
    learning_rate = 0.15;

    constructor(){
        this.w1 = (Math.random() * 2) - 1;
        this.w2 = (Math.random() * 2) - 1;
    }
  
    output(i1,i2) {
      return this.activation(this.w1 * i1 + this.w2 * i2 + this.b); 
    }
  
    activation(x){
        // return 1/(1 + Math.exp(-x));
        // return x;
        if(x > 0.5) return 1;
        else return 0; 
        // if(x > 0) return x;
        // else return 0; 
    }
  
    train(i1,i2,actual_out){
      let y = this.output(i1,i2);
  
      let delta_b  = (actual_out - y) * this.learning_rate;
      let delta_w1 = delta_b * i1;
      let delta_w2 = delta_b * i2;
  
      this.w1 += delta_w1;
      this.w2 += delta_w2;
      this.b  += delta_b;
    }
  
  }


  class MLP {
    constructor(inputSize, hiddenSize, outputSize) {
      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      this.outputSize = outputSize;
  
      // Initialize weights and biases
      this.weights1 = math.random([inputSize, hiddenSize], -1, 1);
      this.biases1 = math.zeros([1, hiddenSize]);
      this.weights2 = math.random([hiddenSize, outputSize], -1, 1);
      this.biases2 = math.zeros([1, outputSize]);
    }
  
    // Sigmoid activation function
    sigmoid(x) {
      return math.divide(1, math.add(1, math.exp(math.multiply(-1, x))));
    }
  
    // Mean squared error loss function
    mse(output, target) {
      return math.mean(math.square(math.subtract(output, target)));
    }
  
    // Forward pass through the network
    forward(input) {
      this.hidden = this.sigmoid(math.add(math.multiply(input, this.weights1), this.biases1));
      this.output = this.sigmoid(math.add(math.multiply(this.hidden, this.weights2), this.biases2));
      return this.output;
    }
  
    // Backward pass through the network
    backward(input, target, learningRate) {
      // Calculate error and delta for output layer
      const outputError = math.subtract(target, this.output);
      const outputDelta = math.dotMultiply(outputError, math.dotMultiply(this.output, math.subtract(1, this.output)));
  
      // Calculate error and delta for hidden layer
      const hiddenError = math.multiply(outputDelta, math.transpose(this.weights2));
      const hiddenDelta = math.dotMultiply(hiddenError, math.dotMultiply(this.hidden, math.subtract(1, this.hidden)));
  
      // Update weights and biases
      this.weights2 = math.add(this.weights2, math.multiply(learningRate, math.multiply(math.transpose(this.hidden), outputDelta)));
      this.biases2 = math.add(this.biases2, math.multiply(learningRate, outputDelta));
      this.weights1 = math.add(this.weights1, math.multiply(learningRate, math.multiply(math.transpose(input), hiddenDelta)));
      this.biases1 = math.add(this.biases1, math.multiply(learningRate, hiddenDelta));
    }
  
    // Train the network on a dataset
    train(dataset, numEpochs, learningRate) {
      for (let i = 0; i < numEpochs; i++) {
        let loss = 0;
        for (let j = 0; j < dataset.length; j++) {
          const input = dataset[j].input;
          const target = dataset[j].target;
          const output = this.forward(input);
          this.backward(input, target, learningRate);
          loss += this.mse(output, target);
        }
        console.log(`Epoch ${i+1}: loss = ${loss/dataset.length}`);
      }
    }
  
    // Make a prediction on a single input
    predict(input) {
      return this.forward(input);
    }
  }
  