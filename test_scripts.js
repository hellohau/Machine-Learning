const C = require('./Classes.js');

function softmax(arr){
    
    let max = Math.max(...arr);
    let exp_sum = (arr.map((x) => {return Math.exp(x - max);})).reduce((a, b) => a + b, 0);

    return arr.map((x) => {return Math.exp(x - max) / exp_sum;});
}

function derivate_softmax1(arr){
    let res = [];
    for(let i = 0; i < arr.length; i++){
        let sum = 0;
        for(let j = 0; j < arr.length; j++){
            if(i == j) sum += arr[i] * (1 - arr[i]);
            else sum += (-arr[i]) * arr[j];
        }

        res.push(sum);
    }

    return res;
} 


function derivate_softmax2(arr){
    let res = [];
    for(let i = 0; i < arr.length; i++){
        let row = [];
        for(let j = 0; j < arr.length; j++){
            if(i == j) row.push(arr[i] * (1 - arr[i]));
            else row.push((-arr[i]) * arr[j]);
        }

        res.push(row);
    }

    return res;
} 

let ta = [1,2,3,4,5];
let t = softmax([1,2,3,4,5]);
// console.log("Softmax : ", t);
// console.log("derivative 1 : ", derivate_softmax1(t));
// console.log("Rederivate : ", (derivate_softmax2(t)).map((x) => x.reduce((a,b) => a+b,0)));
// console.log("derivative 2 : ", derivate_softmax2(t));

console.log("Sum : ", t.reduce((a,b) => a+b,0));

let trans = C.MMath.transpose(derivate_softmax2(t));
let der = derivate_softmax2(t);

console.log("Gradient 2: ",C.MMath.mat_mult(ta,der));
console.log("Gradient 33: ",C.MMath.mat_mult(ta,trans));
console.log("Der : ",der);
console.log("Transposed : ",trans);
console.log("Der : ",der);


// console.log("Gradient 1: ",C.MMath.mat_one_to_one_mult(ta,derivate_softmax1(t)));
