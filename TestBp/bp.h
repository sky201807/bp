﻿#ifndef _BP_H_  
#define _BP_H_  

#include <vector>  

//参数设置
#define LAYER    3        //三层神经网络  
#define NUM      10       //每层的最多节点数  

#define A        30.0  
#define B        10.0     //A和B是S型函数的参数  
#define ITERS    2000     //最大训练次数  
#define ETA_W    0.0035   //权值调整率  
#define ETA_B    0.001    //阀值调整率  
#define ERROR    0.002    //单个样本允许的误差  
#define ACCU     0.005    //每次迭代允许的误差  

//类型
#define Type double  
#define Vector std::vector  

struct Data
{
    Vector<Type> x;       //输入属性
    Vector<Type> y;       //输出属性
};

class BP{

public:

    void GetData(const Vector<Data>);
    void Train();
    Vector<Type> ForeCast(const Vector<Type>);
    void ForCastFromFile(BP * &);
    void ReadFile(const char * InutFileName,int m, int n);

    void ReadTestFile(const char * InputFileName, int m, int n);
    void WriteToFile(const char * OutPutFileName);

private:

    void InitNetWork();         //初始化网络  
    void GetNums();             //获取输入、输出和隐含层节点数  
    void ForwardTransfer();     //正向传播子过程  
    void ReverseTransfer(int);  //逆向传播子过程  
    void CalcDelta(int);        //计算w和b的调整量  
    void UpdateNetWork();       //更新权值和阀值  
    Type GetError(int);         //计算单个样本的误差  
    Type GetAccu();             //计算所有样本的精度  
    Type Sigmoid(const Type);   //计算Sigmoid的值  
    void split(char *buffer, Vector<Type> &vec);

private:
    int in_num;                 //输入层节点数  
    int ou_num;                 //输出层节点数  
    int hd_num;                 //隐含层节点数  

    Vector<Data> data;            //样本数据
    Vector<Vector<Type>> testdata;//测试数据
    Vector<Vector<Type>> result;  //测试结果
    int        rowLen;            //样本数量
    int        restrowLen;        //测试样本数量

    Type w[LAYER][NUM][NUM];    //BP网络的权值  
    Type b[LAYER][NUM];         //BP网络节点的阀值  

    Type x[LAYER][NUM];         //每个神经元的值经S型函数转化后的输出值，输入层就为原值  
    Type d[LAYER][NUM];         //记录delta学习规则中delta的值,使用delta规则来调整联接权重 Wij(t+1)=Wij(t)+α(Yj-Aj(t))Oi(t)
};

#endif  //_BP_H_  