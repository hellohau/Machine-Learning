class MLP {
    
    input_nodes = [];
    hidden_nodes = [];
    output_nodes = [];
    learning_rate = 0.1;
    hidden_activation = "relu";
    output_activation = "sigmoid";

    // ex : 3 , 4 , [4 , 4] 
    constructor(n_inputs,n_outputs,n_hidden_arr = []){
        this.n_inputs = n_inputs;
        this.n_hidden_arr = n_hidden_arr;
        this.n_outputs = n_outputs;

        this.initialize_weights();
        this.initialize_biases();
    }

    activate(x,activation_name) {
        switch(activation_name){
            case "relu":        
                if(Array.isArray(x)) return x.map((y) => {return this.activate(y,activation_name);});
                return x > 0 ? x : 0;
            
            case "sigmoid":
                if(Array.isArray(x)) return x.map((y) => {return this.activate(y,activation_name);});
                return 1/(1 + Math.exp(-x));
            
            case "softmax":
                let max = Math.max(...x);
                let exp_sum = (x.map((y) => {return Math.exp(y - max);})).reduce((a, b) => a + b, 0);
            
                return x.map((y) => {return Math.exp(y - max) / exp_sum;});
            
            default : console.trace("activation function name not found");
                return;
        }

    }
    
    derivate_activate(x,activation_name){

        switch(activation_name){
            case "relu":        
                if(Array.isArray(x)) return x.map((y) => {return this.derivate_activate(y,activation_name);});
                return (x > 0 ? 1 : 0);
            
            case "sigmoid":
                if(Array.isArray(x)) return x.map((y) => {return this.derivate_activate(y,activation_name);});
                return x*(1-x);
            
            case "softmax":
                let res = [];
                for(let i = 0; i < x.length; i++){
                    let row = [];
                    for(let j = 0; j < x.length; j++){
                        if(i == j) row.push(x[i] * (1 - x[i]));
                        else row.push((-x[i]) * x[j]);
                    }
            
                    res.push(row);
                }
            
                return res;
            
            default : console.trace("derivate activation function name not found");
                return;
        }


    }

    initialize_weights(){
        this.weights = [];

        if(this.n_hidden_arr.length == 0){
            this.weights.push(MMath.xavier_init(this.n_inputs,this.n_outputs));

        }
        else{
            this.weights.push(MMath.xavier_init(this.n_inputs,this.n_hidden_arr[0]));

            for(let i = 1; i < this.n_hidden_arr.length; i++){
                this.weights.push(MMath.xavier_init(this.n_hidden_arr[i-1],this.n_hidden_arr[i]));
            }

            this.weights.push(MMath.xavier_init(this.n_hidden_arr[this.n_hidden_arr.length - 1],this.n_outputs));
            
        }

    }

    initialize_biases(){
        this.biases = [];

        for (let i = 0; i < this.n_hidden_arr.length; i++) {
            this.biases.push(MMath.mat_filled_num(1,this.n_hidden_arr[i],0));
        }

        this.biases.push(MMath.mat_filled_num(1,this.n_outputs,0));

    }

    feedforward(inputs){
        this.input_nodes = inputs;
        this.hidden_nodes = [];

        if(this.n_hidden_arr.length > 0){
            
            this.hidden_nodes.push(this.activate(
                MMath.mat_add(
                    MMath.mat_mult(this.input_nodes,this.weights[0]),
                    this.biases[0])
                ,this.hidden_activation)
            );
            
            for (let i = 1; i < this.n_hidden_arr.length; i++) {
                   
                this.hidden_nodes.push(this.activate(
                    MMath.mat_add(
                        (MMath.mat_mult(this.hidden_nodes[this.hidden_nodes.length - 1],this.weights[i])),
                        this.biases[i])
                    ,this.hidden_activation)
                );         

            }

            this.output_nodes = 
            this.activate(MMath.mat_add(
                    MMath.mat_mult(this.hidden_nodes[this.hidden_nodes.length - 1],this.weights[this.weights.length - 1]),
                    this.biases[this.biases.length-1]
                ),this.output_activation);

        }
        else{
            this.output_nodes = this.activate(MMath.mat_add(MMath.mat_mult(this.input_nodes,this.weights[0]),this.biases[0]),this.output_activation);
        }
        
        return this.output_nodes;
    }

    backprop(actual_outputs){

        let bias_deltas   = [];
        let weight_deltas = [];

        // dL/da * da/dz
        let output_error = null;
        if(this.output_activation == "softmax"){
            output_error = MMath.mat_mult(
                MMath.mat_mult_num(
                    MMath.mat_sub(this.output_nodes,actual_outputs)
                    ,2),
                this.derivate_activate(this.output_nodes,this.output_activation)
            );
        }else{
            output_error = MMath.mat_one_to_one_mult(
                MMath.mat_mult_num(
                    MMath.mat_sub(this.output_nodes,actual_outputs)
                    ,2),
                this.derivate_activate(this.output_nodes,this.output_activation)
            );
        }

        if(output_error == null) {console.trace("error, output_error not computed");throw("output_error");}

        
        bias_deltas.unshift(output_error);

        //Take into account the hidden layer
        if(this.n_hidden_arr.length > 0){

            let curr_error = output_error;

            for(let i = this.n_hidden_arr.length - 1; i >= 0; i--){

                weight_deltas.unshift(MMath.mat_mult(
                    MMath.transpose(this.hidden_nodes[i]),
                    curr_error
                ));
                
                curr_error = MMath.mat_one_to_one_mult(
                    MMath.mat_mult(curr_error , MMath.transpose(this.weights[i + 1])),
                    this.derivate_activate(this.hidden_nodes[i],this.hidden_activation)
                );
                bias_deltas.unshift(curr_error);

            }

            weight_deltas.unshift(MMath.mat_mult(
                MMath.transpose(this.input_nodes),
                curr_error
            ));

        }
        else{

            weight_deltas.unshift(MMath.mat_mult(
                MMath.transpose(this.input_nodes),
                output_error
            ));

        }

        // //Update the weights/biases with the deltas
        for(let i = 0; i < this.weights.length; i++){

            this.weights[i] = MMath.mat_sub(this.weights[i],MMath.mat_mult_num(weight_deltas[i],this.learning_rate));
            this.biases[i] = MMath.mat_sub(this.biases[i],MMath.mat_mult_num(bias_deltas[i],this.learning_rate));
        }

    }



}

class MMath{

    static dot_product(m1,m2){
        
        //Test if the computation is possible
        if(m1.length != m2.length) {console.trace(m1,m2); throw("Different Array sizes ");}

        let sum = 0;
        
        for(let i = 0; i < m1.length; i++){
            sum += m1[i] * m2[i];    
        }

        return sum;
    };

    // if(Array.isArray(m1[0]) && Array.isArray(m2[0])) return this.dot_product(m1,m2); 

    static mat_mult(m1,m2){

        if(!Array.isArray(m1[0])) m1 = [m1];
        if(!Array.isArray(m2[0])) m2 = [m2];

        if(m1[0].length != m2.length) {console.trace(m1,m2);throw("Wrong Array sizes");}

        let res = [];

        //Test if the computation is possible
        for(let i = 0; i < m1.length; i++){
            let row = [];
            for(let j = 0; j < m2[0].length; j++){
                

                let sum = 0;
                
                for(let k = 0 ; k < m1[0].length; k++){
                    sum += m1[i][k] * m2[k][j];
                }    

                row.push(sum);
            }

            res.push(row);
        }

        if(res.length == 1 && Array.isArray(res[0])) res = res[0];
        return res;
    }

    static mat_add(m1,m2,add = 1){
        
        if(!Array.isArray(m1[0])) m1 = [m1];
        if(!Array.isArray(m2[0])) m2 = [m2];
        
        if(m1.length != m2.length || m1[0].length != m2[0].length) {console.trace(m1,m2);throw("Wrong size arrays")};

        let res = [];
        for (let i = 0; i < m1.length; i++) {
            let row = [];
            for (let j = 0; j < m1[0].length; j++) {
                if(add) row.push(m1[i][j] + m2[i][j]);
                else row.push(m1[i][j] - m2[i][j]);
            }

            res.push(row);
        }

        if(res.length == 1 && Array.isArray(res[0])) res = res[0];
        return res;
    }

    static mat_sub(m1,m2){
        return this.mat_add(m1,m2,0);
    }

    static mat_add_num(m1,num,add = 1){
        
        
        if(!Array.isArray(m1[0])) m1 = [m1];
        
        if(m1 == undefined || num == undefined || num == NaN) {console.trace(m1,num);throw("Error Matrix add num");}

        let res = [];
        for (let i = 0; i < m1.length; i++) {
            let row = [];
            for (let j = 0; j < m1[0].length; j++) {
                if(add) row.push(m1[i][j] + num);
                else row.push(m1[i][j] - num);
            }

            res.push(row);
        }

        if(res.length == 1 && Array.isArray(res[0])) res = res[0];
        return res;
    }

    static mat_sub_num(m1,num){
        return this.mat_add_num(m1,num,0);
    }

    static mat_mult_num(m1,num){
        if(!Array.isArray(m1[0])) m1 = [m1];

        let res = [];
        for(let i = 0; i < m1.length; i++){
            let row = [];
            for(let j = 0; j < m1[0].length; j++){
                row.push(m1[i][j] * num);
            }

            res.push(row);
        }

        if(res.length == 1 && Array.isArray(res[0])) res = res[0];
        return res;
    }
    
    static rand_matrix(row,col){

        if(row <= 0 || col <= 0) throw("Wrong row,col");

        let res = [];
        for(let i = 0; i < row; i++){
            let row = [];
            for(let j = 0; j < col; j++){
                row.push(Math.random());
            }

            res.push(row);
        }

        return res;
    }

    static mat_filled_num(row,col,num){
        
        if(row <= 0 || col <= 0) throw("Wrong row,col");
        
        let res = [];
        for(let i = 0; i < row; i++){
            let row = [];
            for(let j = 0; j < col; j++){
                row.push(num);
            }

            res.push(row);
        }

        return res;
    }

    static xavier_init(n_inputs, n_outputs) {
        // const limit = Math.sqrt(6 / (n_inputs + n_outputs));
        const limit = Math.sqrt(1 / (n_inputs));
        const res = [];
      
        for (let i = 0; i < n_inputs; i++) {
          let row = []
          for (let j = 0; j < n_outputs; j++) {
            row.push((2 * Math.random() - 1) * limit);
          }

          res.push(row);
        }
      
        return res;
    }

    static transpose(mat){

        if(!Array.isArray(mat[0])) mat = [mat];
        
        let result = [];
        
        for(let i = 0; i < mat[0].length; i++){
            let row = [];
            for(let j = 0; j < mat.length; j++){
                row.push(mat[j][i]);
            }

            result.push(row);
        }

        if(result.length == 1 && Array.isArray(result[0])) result = result[0];

        return result;
    }

    static mat_one_to_one_mult(m1,m2){

        if(!Array.isArray(m1[0])) m1 = [m1];
        if(!Array.isArray(m2[0])) m2 = [m2];
        
        if(m1.length != m2.length || m1[0].length != m2[0].length) {console.trace(m1,m2); throw("Matrices are different sizes for 1 to 1 mult");}

        let res = [];
        for (let i = 0; i < m1.length; i++) {
            let row = [];
            for (let j = 0; j < m1[0].length; j++) {
                row.push(m1[i][j] * m2[i][j]);
            }
            
            res.push(row);
        }

        if(res.length == 1 && Array.isArray(res[0])) res = res[0];

        return res;
    }

    static diag(mat){

        let res = [];
        
        for(let i = 0; i < mat.length; i++){
            let row = [];
            for(let j = 0; j < mat.length; j++){
                if(i == j) row.push(mat[i]);
                else row.push(0); 
            }

            res.push(row);
        }

        return res;
    }
}

module.exports = {MLP,MMath};