const Classes = require('./Classes.js');
const fs = require('fs');

let m = new Classes.MLP(4,3,[3]);
m.learning_rate = 0.1;

function train(dataset,mlp,min_cost){
    let iteration = 0;
    let cost = 1;
    while(cost > min_cost) {
        let sqr_sum = 0;
        for(let j = 0; j < dataset.length; j++){
            let out = mlp.feedforward(dataset[j].x);
            mlp.backprop(dataset[j].y);

            sqr_sum += ((Classes.MMath.mat_sub(out,dataset[j].y)).map((x) => {return x*x;})).reduce((a,b) => a+b,0);
        }

        cost = sqr_sum/dataset.length;
        console.log("Cost %d : ", iteration++,cost);
    }
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

fs.readFile('../TestingStuff/iris.data','utf8',(err,data) => {
    if(err){console.log(err);return;}

    let test_data = data.split('\n');

    for (let i = 0; i < test_data.length; i++) {
        let test_arr = test_data[i].split(',');
        let last = test_arr.splice(-1,1)[0];
        switch(last){
            case 'Iris-setosa': last = [1,0,0];
                break;
            case 'Iris-versicolor': last = [0,1,0];
                break;
            case 'Iris-virginica': last = [0,0,1];
                break;
            default : console.log("No correspondance",last);
                break;
        }

        test_data[i] = {x:test_arr.map((x) => {return parseFloat(x);}), y:last};
    }

    let training_data = pick_random(test_data,125);

    train(training_data,m,0.1);
    test(test_data,m,0.1);

    // console.log(m);
});

function pick_random(arr,num){
    if(num > arr.length){console.log("Number bigger than array");return;}

    let res = []; 
    for(let i = 0; i < num; i++){
        let index = Math.floor(Math.random() * arr.length);
        res.push(arr.splice(index,1)[0]);
    }

    return res;
}



