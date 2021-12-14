package com.company;

public class Helper {

        public static double[][] multiplicacion(double[][] a, double[][] b) {
            double[][] respuesta=new double[a.length][b[0].length];
            for(int i = 0; i<respuesta.length; i++)
            {
                for(int j = 0; j<respuesta[0].length; j++)
                {
                    double sum=0;
                    for(int k = 0; k<a[0].length; k++)
                    {
                        sum+=a[i][k]*b[k][j];
                    }
                    respuesta[i][j]=sum;
                }
            }
            return respuesta;
        }
        public static double[][] mat_Sum(double[][] mat_A, double[][] mat_B){
            double[][] mat_sum;

            if(mat_A.length == mat_B.length && mat_B[0].length == mat_A[0].length){
                mat_sum = new double[mat_A.length][mat_A[0].length];
                for(int i=0;i<mat_A.length;i++){
                    for(int j=0;j<mat_A[0].length;j++){
                        mat_A[i][j] +=mat_B[i][j];
                    }
                }
                return mat_A;
            } else {
                return mat_A;
            }

        }
        public static double[][] sigmoide(double [][] neuron) {
            for(int i = 0; i < neuron.length; i++)
            {
                for(int j = 0; j < neuron[0].length; j++)
                    neuron[i][j] = 1/(1+Math.exp(-neuron[i][j]));
            }
            return neuron;
        }

        public static double[][] mat_rest(double[][] mat_A, double[][] mat_B){
            double[][] mat_rest;

            if(mat_A.length == mat_B.length && mat_B[0].length == mat_A[0].length){
                mat_rest = new double[mat_A.length][mat_A[0].length];
                for(int i=0;i<mat_A.length;i++){
                    for(int j=0;j<mat_A[0].length;j++){
                        mat_rest[i][j] = mat_A[i][j] - mat_B[i][j];
                    }
                }
                return mat_rest;
            } else {
                return mat_A;
            }
        }

        public static double[][] derivada_sigmoide(double[][] ma) {
            double[][] ans=new double[ma.length][ma[0].length];
            for(int i = 0; i< ma.length; i++)
            {
                for(int j = 0; j< ma[0].length; j++)
                    ans[i][j] = ma[i][j] * (1-ma[i][j]);
            }
            return ans;
        }

        public static double[][] multi(double[][] ma, double a) {
            double[][] ans=new double[ma.length][ma[0].length];

            for(int i = 0; i< ma.length; i++)
            {
                for(int j = 0; j< ma[0].length; j++)
                {
                    ans[i][j] = ma[i][j] *a;
                }
            }
            return ans;
        }

        public static double [][] matrix_XT (double [][] matrix_X){
            double [][] mat_XT = new double[matrix_X[0].length][matrix_X.length];
            for (int i = 0; i < matrix_X.length; i++) {
                for (int j = 0; j < matrix_X[0].length; j++){
                    mat_XT[j][i] = matrix_X[i][j];

                }
            }
            return mat_XT;
        }

        public static void ImprimirMatriz(double[][] m){
            for (int i = 0; i < m.length; i++) { //filas
                for (int j = 0; j < m[i].length ; j++) { //columnas
                    System.out.print(m[i][j] + " ");

                }
                System.out.println("");
            }
        }

        public static double[][] convertir_Array_Matrix(double[]x) {

            double[][] result = new  double[x.length][1];
            for(int i =0;i<x.length;i++)
                result[i][0]=x[i];
            return result;
        }
        public static double[][] de3dto2d(double[][][] matriz, int capa){
            double[][] resultado = new double[matriz.length][matriz[0].length];

            for (int i = 0; i <resultado.length ; i++) {
                for (int j = 0; j < resultado[0].length ; j++) {
                    resultado[i][j]= matriz[i][j][capa];
                }
            }
            return resultado;


        }

        public static double[][][] remplece2dto3d(double[][][] matriz, double[][] ma, int capa){

            for (int i = 0; i <ma.length ; i++) {
                for (int j = 0; j < ma[0].length ; j++) {
                    matriz[i][j][capa]= ma[i][j];
                }
            }
            return matriz;
        }

    public static double [][] multiplicacion2(double [][] a, double [][] b) {
        for(int i = 0; i<b.length; i++)
        {
            for(int j = 0; j<b[0].length; j++)
            {
                a[i][j]*=b[i][j];
            }
        }
        return a;
    }


}
