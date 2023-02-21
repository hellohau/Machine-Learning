class MLP {
    
    // ex : 3 , [4 , 4] , 4
    constructor(n_inputs,n_outputs,n_hidden_arr = []){
        this.n_inputs = n_inputs;
        this.n_hidden_arr = n_hidden_arr;
        this.n_outputs = n_outputs;

        this.initialize_weights();
    }

    activate(x) {return 1/(1 + Math.exp(-x));}

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

    
}

class MMath{

    static dot_product(m1,m2){
        
        //Test if the computation is possible
        if(m1.length != m2.length) throw("Different Array sizes ",m1,m2);

        let sum = 0;
        
        for(let i = 0; i < m1.length; i++){
            sum += m1[i] * m2[i];    
        }

        return sum;
    };

    // if(Array.isArray(m1[0]) && Array.isArray(m2[0])) return this.dot_product(m1,m2); 

    static mat_mult(m1,m2){
        
        if(m1[0].length != m2.length) throw("Wrong Array sizes",m1,m2);

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

            if(row.length == 1) row = row[0];

            res.push(row);
        }

        if(res.length == 1) res = res[0];

        return res;
    }

    static mat_add(m1,m2,add = 1){
        if(m1.length != m2.length || m1[0].length != m2[0].length) throw("Wrong size arrays",m1,m2);

        let res = [];
        for (let i = 0; i < m1.length; i++) {
            let row = [];
            for (let j = 0; j < m1[0].length; j++) {
                if(add) row.push(m1[i][j] + m2[i][j]);
                else row.push(m1[i][j] - m2[i][j]);
            }

            if(row.length == 1) row = row[0];
            res.push(row);
        }

        return res;
    }

    static mat_sub(m1,m2){
        return this.mat_add(m1,m2,0);
    }

    static mat_add_num(m1,num,add = 1){
        
        if(m1 == undefined || num == undefined || num == NaN) throw("Error Matrix add num",m1,num);

        let res = [];
        for (let i = 0; i < m1.length; i++) {
            let row = [];
            for (let j = 0; j < m1[0].length; j++) {
                if(add) row.push(m1[i][j] + num);
                else row.push(m1[i][j] - num);
            }

            if(row.length == 1) row = row[0];
            res.push(row);
        }

        return res;
    }

    static mat_sub_num(m1,num){
        return this.mat_add_num(m1,num,0);
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

    static zero_matrix(row,col){
        
        if(row <= 0 || col <= 0) throw("Wrong row,col");
        
        let res = [];
        for(let i = 0; i < row; i++){
            let row = [];
            for(let j = 0; j < col; j++){
                row.push(0);
            }

            res.push(row);
        }

        return res;
    }

    static xavier_init(n_inputs, n_outputs) {
        const limit = Math.sqrt(6 / (n_inputs + n_outputs));
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
}