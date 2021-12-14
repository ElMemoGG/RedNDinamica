package com.company;

public class Red_neurona {

    double [][][] weights1;
    double [][] weights2;
    double [][][] biasH;
    double [][] biasO;
    double alpha = 0.01;
    int NeuronaS;
    double[][] hidden;


    public Red_neurona(int inputS, int hiddenS, int ouputS, int neuroS) {
        NeuronaS = neuroS;
        weights1 = inicializar(hiddenS,inputS,neuroS);
        weights2 = inicializar(ouputS,hiddenS);
        biasH = inicializar(hiddenS, 1, neuroS);
        biasO = inicializar(ouputS, 1);
    }

    public double[][] prediccion(double[] x){
        hidden= Helper.convertir_Array_Matrix(x);
        for (int i = 0; i <NeuronaS; i++) {
            hidden = hacerprediccion(x, Helper.de3dto2d(weights1, i),Helper.de3dto2d(biasH, i) );
        }
        double[][] output = Helper.multiplicacion(weights2, hidden);
        output = Helper.mat_Sum(output, biasO);
        output = Helper.sigmoide(output);
        return output;
    }

    public double[][] hacerprediccion(double[] input, double[][] weight, double[][] bias ){
        double[][] input_user= Helper.convertir_Array_Matrix(input);
        double[][] hidden = Helper.multiplicacion(weight, input_user);
        hidden = Helper.mat_Sum(hidden, bias);
        hidden = Helper.sigmoide(hidden);
        return hidden;
    }


    public void train(double[] x, double[] y){

        double[][] target = Helper.convertir_Array_Matrix(y);
        double[][] input_user =Helper.convertir_Array_Matrix(x);

        //Optenemos el output
        double[][] output =  prediccion(x);

        //obtenemos la gradiente
        double[][] error = Helper.mat_rest(target, output);
        double[][] gradiente = Helper.derivada_sigmoide(output);
        gradiente = Helper.multiplicacion2(gradiente, error);
        gradiente = Helper.multi(gradiente,alpha);

        double[][] who_delta = Helper.multiplicacion( gradiente, Helper.matrix_XT(hidden));
        weights2 = Helper.mat_Sum(weights2, who_delta);
        biasO = Helper.mat_Sum(biasO, gradiente);

        //de aqui en adelante quien sabe que vergas hace esto y como se pueda hacer dinamico es un misterio


        for (int i = NeuronaS-1; i >= 0 ; i--) {
            backpropagation(who_delta, error,input_user, i);
        }

    }
    public void backpropagation(double[][] delta, double[][]error,double[][]input_user, int capa){

        double[][] weightT = Helper.de3dto2d(weights1, capa);
        double[][] biasT = Helper.de3dto2d(biasH, capa);

        double[][] hidden_errors = Helper.multiplicacion(Helper.matrix_XT(delta), error);

        double[][] h_gradiente = Helper.derivada_sigmoide(hidden);
        h_gradiente = Helper.multiplicacion2(h_gradiente, hidden_errors );
        h_gradiente = Helper.multi(h_gradiente, alpha);

        double[][] wih_delta = Helper.multiplicacion( h_gradiente, Helper.matrix_XT(input_user));

        weightT = Helper.mat_Sum(weightT, wih_delta);
        biasT = Helper.mat_Sum(biasT, h_gradiente);

        weights1 = Helper.remplece2dto3d(weights1, weightT, capa);
        biasH = Helper.remplece2dto3d(biasH, biasT, capa);

    }



    public void cycleTraining(double[][]x, double[][]y, int iteration)
    {
        for(int i=0;i<iteration;i++)
        {
            int sampleN =  (int)(Math.random() * x.length );
            train(x[sampleN], y[sampleN]);
        }
    }



    public double[][][] inicializar(int filas, int columnas,int capas ){
        double[][][] result = new double[filas][columnas][capas];
        for (int k = 0; k < capas ; k++) {
            for (int i = 0; i < filas; i++) {
                for (int j = 0; j < columnas; j++) {
                    result[i][j][k] =Math.random()*2-1;
                    //result[i][j][k] = 1;
                }
            }
        }
        return  result;
    }

    public double[][] inicializar(int filas, int columnas ){
        double[][] result = new double[filas][columnas];
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                result[i][j] =Math.random()*2-1;
                //result[i][j] = 1;
            }
        }
        return  result;
    }
}
