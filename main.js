let m = new MLP(2,1,[4]);
m.learning_rate = 0.2;

// let dataset = [{x:[0,0], y:[0]},{x:[1,0], y:[0]},{x:[0,1], y:[0]},{x:[1,1], y:[1]}];
// let dataset = [{x:[0,0], y:[0]},{x:[1,0], y:[1]},{x:[0,1], y:[1]},{x:[1,1], y:[1]}];
let dataset = [{x:[0,0], y:[0]},{x:[1,0], y:[1]},{x:[0,1], y:[1]},{x:[1,1], y:[0]}];

function train(){
    let iteration = 0;
    let cost = 1;
    while(cost > 0.01) {
        let sqr_sum = 0;
        for(let j = 0; j < dataset.length; j++){
            let out = m.feedforward(dataset[j].x);
            m.backprop(dataset[j].y);

            sqr_sum += Math.pow(out - dataset[j].y,2);
        }

        cost = sqr_sum/4;
        // console.log("Cost %d : ", iteration++,cost);
    }
    
    console.log("[0,0] : ",m.feedforward([0,0]));
    console.log("[1,0] : ",m.feedforward([1,0]));
    console.log("[0,1] : ",m.feedforward([0,1]));
    console.log("[1,1] : ",m.feedforward([1,1]));
}

// console.log("[0,0] : ",m.feedforward([0,0]));
// console.log("[1,0] : ",m.feedforward([1,0]));
// console.log("[0,1] : ",m.feedforward([0,1]));
// console.log("[1,1] : ",m.feedforward([1,1]));

train();
