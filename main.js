const Classes = require('./Classes.js');
const fs = require('fs');

let m = new Classes.MLP(4,3,[6]);
m.learning_rate = 0.005;
m.hidden_activation = "relu";
m.output_activation = "softmax";

function train(dataset,mlp,min_cost){
    let iteration = 0;
    let cost = 1;
    while(cost > min_cost && iteration < 300) {
        let sqr_sum = 0;
        for(let j = 0; j < dataset.length; j++){
            let out = mlp.feedforward(dataset[j].x);
            mlp.backprop(dataset[j].y);

            sqr_sum += ((Classes.MMath.mat_sub(out,dataset[j].y)).map((x) => {return x*x;})).reduce((a,b) => a+b,0);
        }

        iteration++
        
        cost = sqr_sum/dataset.length;
        // console.log("Cost %d : ", iteration,cost);
    }

    return iteration;
}

function test(dataset,mlp,min_diff){
    console.log("TESTING : ");
    let count = 0;
    for(let i = 0; i < dataset.length; i++){
        let out = mlp.feedforward(dataset[i].x);
        let sum = (Classes.MMath.mat_sub(out,dataset[i].y)).reduce((a,b) => Math.abs(a) + Math.abs(b),0);
        if(sum/dataset.length < min_diff) count++;
        console.log("Prediction : ",out,"Expected",dataset[i].y);
    }

    console.log("Min error %f : Solved %d / %d",min_diff,count,dataset.length);
}

// let dataset = [{x:[0,0],y:[0,1]},{x:[1,0],y:[1,0]},{x:[0,1],y:[1,0]},{x:[1,1],y:[0,1]}];
// let dataset = [{x:[0,0],y:[0]},{x:[1,0],y:[1]},{x:[0,1],y:[1]},{x:[1,1],y:[0]}];

// m.hidden_activation = "sigmoid"
// m.output_activation = "softmax";

// train(dataset,m,0.001);
// test(dataset,m,0.1);

// fs.readFile('../TestingStuff/CADJPY_historical_data 2020-2023.csv','utf8',(err,data) => {
//     if(err){console.log(err);return;}

//     let test_data = data.split('\n');
//     test_data = test_data.map((x) => {return (((x.split(' ')[1]).split(',')).splice(1,5)).map((y => { return parseFloat(y);})) ;});

//     test_data = test_data.map((x) => {
//         let last = x.splice(-1,1);
//         return {x:x,y:last};
//     })

//     let training_data = pick_random(test_data,0.8);

//     train(training_data,m,0.1);
//     test(test_data,m,0.1);

//     console.log(m);
// });

fs.readFile('../Machine-Learning/iris.data','utf8',(err,data) => {
    if(err){console.log(err);return;}

    let test_data = data.split('\n');

    for (let i = 0; i < test_data.length; i++) {
        let test_arr = test_data[i].split(',');
        let last = (test_arr.splice(-1,1)[0]).replace(/(\r\n|\n|\r)/gm, "");;
        switch(last){
            case 'Iris-setosa': last = [1,0,0];
                break;
            case 'Iris-versicolor': last = [0,1,0];
                break;
            case 'Iris-virginica': last = [0,0,1];
                break;
            default : console.log("No correspondance",[last]);
                break;
        }

        test_data[i] = {x:test_arr.map((x) => {return parseFloat(x);}), y:last};
    }

    let training_data = pick_random(test_data,0.9);
    // train(training_data,m,0.1);
    // test(test_data,m,0.1);

    let count = 0;
    let solve_count = 0;
    for (let j = 0; j < 500; j++) {;
        let m = new Classes.MLP(4,3,[6]);
        m.learning_rate = 0.0069;
        m.hidden_activation = "relu"
        m.output_activation = "softmax";

        let it = train(training_data,m,0.1);
        console.log("Iterations : ",it);
        if(it < 300) {count++;solve_count+= it;}
        // test(test_data,m,0.1);
    }

    console.log("Count succeeded : %d / %d",count,500);
    console.log("Average iteration : %d",solve_count/500);


    // console.log(m);
});

function pick_random(arr,percent){
    if(percent > 1){console.log("Number bigger than array");return;}

    let num = Math.ceil(percent * arr.length);
    let res = []; 
    for(let i = 0; i < num; i++){
        let index = Math.floor(Math.random() * arr.length);
        res.push(arr.splice(index,1)[0]);
    }

    return res;
}
