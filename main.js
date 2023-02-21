let a = [[1,2,3],[4,5,6]];
// let b = [[7,8],[9,10],[11,12]];
let b = [[1,2,3],[4,5,6]];

let m = new MLP(2,2,[3,3]);

// console.log(MMath.xavier_init(3,4));
m.feedforward([1,1]);
// console.log(m);

console.log(MMath.mat_mult(MMath.transpose([1,2]),[[1,2,3]]))
// console.log(MMath.transpose([1,2,3]));

