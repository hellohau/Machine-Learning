// const { MLP, MMath } = require('./Classes.js');

let t = new MLP(1,[4],1);

let dataset = [{x : [0,0] ,y: [0]},{x : [1,0] ,y: [1]},{x : [0,1] ,y: [1]},{x : [1,1] ,y: [0]}];

t.backprop([1,1,1],[1,1,1]);


