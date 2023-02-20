// Create a new MLP with 2 inputs, 4 hidden nodes, and 1 output
const mlp = new MLP(2, 4, 1);

// Define a dataset
const dataset = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

// Train the MLP for 1000 epochs with a learning rate of 0.1
for (let i = 0; i < 1000; i++) {
  for (const data of dataset) {
    const input = data.input;
    const target = data.output;
    mlp.forward(input);
    mlp.backward(input, target, 0.1);
  }
}

// Test the MLP on some inputs
console.log(mlp.forward([0, 0])); // Expected output: [0]
console.log(mlp.forward([0, 1])); // Expected output: [1]
console.log(mlp.forward([1, 0])); // Expected output: [1]
console.log(mlp.forward([1, 1])); // Expected output: [0]