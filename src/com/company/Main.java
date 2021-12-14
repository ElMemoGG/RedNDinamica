package com.company;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
	// write your code here
        double[][] X = {
                {0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1},//IZQUIERDA
                {0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1},//IZQUIERDA
                {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1},//IZQUIERDA
                {0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1},//IZQUIERDA
                {0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0},//DERECHA
                {0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0},//DERECHA
                {0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0}//DERECHA

        };
        double[][] Y = {
                {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}
        };


        Red_neurona nn = new Red_neurona(25, 12, 2, 3);

        nn.cycleTraining(X, Y, 500000);

        double[][] resultado = nn.prediccion(new double[]{1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0});
        System.out.println(new StringBuilder().append("BIas").append(Arrays.deepToString(resultado)).toString());

        //Helper.ImprimirMatriz(resultado);
    }
}
