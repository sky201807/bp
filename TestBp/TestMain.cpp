#include <iostream>  
#include <string.h>  
#include <stdio.h>  
using namespace std;

#include "Bp.h"  

int main()
{
    unsigned int Id, Od;    //样本数据的输入维数/输出维数
    int select = 0;
    BP *bp = new BP();
    const char * inputDataName = "exercisedata.txt";//训练数据文件名称
    const char * testDataName = "testdata.txt";   //测试数据文件名称
    const char * outputDataName = "result.txt";     //输出文件名称

    printf("please input sample input dimension and output dimension:\n");
    scanf("%d,%d", &Id, &Od);
    bp->ReadFile(inputDataName,Id,Od);

    //exercise
    bp->Train();
    //Test
    printf("\n******************************************************\n");
    printf("*1.使用测试文件中国的数据测试 2.从控制台输入数据测试  \n");
    printf("******************************************************\n");
    scanf("%d", &select);
    switch (select)
    {
    case 1:
        bp->ReadTestFile(testDataName,Id,Od);
        bp->ForCastFromFile(bp);
        bp->WriteToFile(outputDataName);
        printf("the result have been save in the file :result.txt.\n");
        break;
    case 2:
        printf("\n\nplease input the Test Data(3 dimension )：\n");
        while (1)
        {
            Vector<Type> in;
            for (int i = 0; i < Id; i++)
            {
                Type v;
                scanf_s("%lf", &v);
                in.push_back(v);
            }
            Vector<Type> ou;
            ou = bp->ForeCast(in);
            printf("%lf\n", ou[0]);
        }
        break;
    default:
        printf("Input error!");
        exit(0);
    }

    return 0;
}