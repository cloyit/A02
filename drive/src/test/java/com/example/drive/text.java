package com.example.drive;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class text {

    public static int[][] zero(int[][] input){
        //检测是否为零，一旦检测到直接返回一个全零
        int [][] result = new int[input.length][input[0].length];
        for(int i=0;i<input.length;i++){
            for (int j = 0; j<input[i].length;j++){
                if(input[i][j]==0){
                    return result;
                }
            }

        }
        //到这里说明没有零的
        return input;
    }
    public static String getMail(String input){
        //观察可知，email 由<>包裹
        String pattern = "<(.*?)>(.*)";

        // 创建 Pattern 对象
        Pattern r = Pattern.compile(pattern);

        Matcher m = r.matcher(input);
        String result = null;
        if(m.find()){
            result = m.group(0);
            //剪切下
           return m.group(0).substring(1,result.length()-1);

        }
        else return null;
    }
 public static void main(String[] args) {
        int[][] notZero  = new int[4][4];
        int[][] haveZero = new int[4][4];
        haveZero[2][3] = 5;
     for(int i = 0; i<notZero.length;i++){
         System.out.println("");
         for (int j = 0; j <notZero[i].length;j++){
             notZero[i][j] = 3;
         }
     }
        int[][] result = null;
        notZero[2][3]= 4;
        System.out.println("含有为0的数组处理后\n");
        result = zero(haveZero);
        for(int i = 0; i<result.length;i++){
            System.out.println("");
            for (int j = 0; j <result[i].length;j++){
                System.out.print(result[i][j]+" ");
            }
        }
        System.out.println("不为0的数组处理后\n");
     result = zero(notZero);
     for(int i = 0; i<result.length;i++){
         System.out.println("");
         for (int j = 0; j <result[i].length;j++){
             System.out.print(result[i][j]+" ");
         }
     }


        System.out.println("");
        String s ="From: \"MR. JAMES NGO\n" +
                "A.\" <jiames ngola2002@makto.com>";
        System.out.println("输入的原始邮箱\n"+s);
        System.out.println("匹配后的邮箱\n"+getMail(s));

    }
}
