
// const training_data = [{i1 : 0, i2 : 0,y : 0},{i1 : 0, i2 : 1,y : 0},{i1 : 1, i2 : 0,y : 0},{i1 : 1, i2 : 1,y : 1}];

// function pick_random_one(arr){
//   let rand_i = Math.floor(Math.random() * arr.length);
//   return arr[rand_i];
// }

// function test_lr(percept){
//   // console.log("Start training");

//   let iteration = 0
//   let error = 1;

//   while(error != 0 && iteration < 100000){
//     error = 0;

//     let data = pick_random_one(training_data);
//     percept.train(data.i1,data.i2,data.y);

//     for (let i = 0; i < training_data.length; i++) {
//       let test_data = training_data[i];
//       if(Math.abs(test_data.y - percept.output(test_data.i1,test_data.i2)) > 0.001){
//         error += 1;
//       }
//     }

//     iteration++;
//   }

  
//   // console.log("Done training in : %i",iteration);
  
//   // console.log("Output (0,0) : %f",p.output(0,0));
//   // console.log("Output (1,0) : %f",p.output(1,0));
//   // console.log("Output (0,1) : %f",p.output(0,1));
//   // console.log("Output (1,1) : %f",p.output(1,1));

//   return iteration;
// }

// let p = new Perceptron();

// for (let i = 0; i < 20; i++) {
//   let avg_it = 0;
//   for (let j = 0; j < 100; j++) {
   
//     p = new Perceptron();
//     p.learning_rate = (i + 1)/20;
//     avg_it += test_lr(p);
//   }

//   avg_it /= 100;

//   console.log("Learning rate : %f \n Average iteration : %i",p.learning_rate, avg_it);
// }

// Define the XOR dataset
const dataset = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] }
];

// Define the network architecture
const inputSize = 2;
const hiddenSize = 4;
const outputSize = 1;

// Create the network
const net = new MLP(inputSize, hiddenSize, outputSize);

// Train the network
const numEpochs = 10000;
const learningRate = 0.1;
net.train(dataset, numEpochs, learningRate);

// Test the network on some example inputs
console.log(net.predict([0, 0]));  // [0.005]
console.log(net.predict([0, 1]));  // [0.991]
console.log(net.predict([1, 0]));  // [0.986]
console.log(net.predict([1, 1]));  // [0.015]
