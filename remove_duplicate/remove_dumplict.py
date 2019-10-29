#!/usr/bin/python  
#-*- coding:utf-8 -*-  
############################  
#File Name: remove_dumplict.py
#Author: dengbf 
#Mail: dengbf@rd.netease.com  
#Created Time: 2019-10-14 22:19:57
############################  

import argparse
import math 

def getdist(x1,y1,x2,y2):
    #print(x1,y1,x2,y2)
    dist = math.sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2))
    return dist


def removedumplict(prdlist,resultlist):
    # 1.读取整个list
    wordDump = 0
    with open(prdlist) as fr, open(resultlist,'w') as w_prd:
        for lineId, line in enumerate(fr.readlines()):
            if (lineId % 100 ==0):
                print(lineId)
            if lineId == 0:
                w_prd.write(line)
                continue
            line = line.strip()
            conts = line.split(",")
            imgName = conts[0]
            row_prd = imgName + ","
            results = conts[1].split(" ")
            if len(results) <= 1:
                ##TODO 直接写入整个结果
                w_prd.write(row_prd+"\n")
                continue 
            #print(len(results))
            #print(line)
            #print(results)
            assert(len(results) %3 == 0)
            words_l = results[0::3] 
            words_x = [int(x) for x in results[1::3]]
            words_y = [int(y) for y in results[2::3]]
            paired = [0] * len(words_x)
            assert(len(words_l) == len(words_x) and len(words_x) == len(words_y)) 
            # 2.读取每一个点
            rmcount = 0
            for i in range(len(words_l)):
                if paired[i]:
                    continue
                for j in range(i+1,len(words_l)):
                    # 判断是否字符相等并且距离小于二十个像素 
                    if words_l[i] == words_l[j] and getdist(words_x[i],words_y[i],words_x[j],words_y[j]) < 20:
                        # 将成对的点进行合并 
                        wordDump += 1
                        paired[i] = 1
                        paired[j] = 1
                        print(imgName, words_l[i],words_l[j],words_x[i],words_y[i],words_x[j],words_y[j])
                        rmcount += 1
                        row_prd += words_l[i]+ " " + str(int((words_x[i] + words_x[j])/2)) + " " + str(int((words_y[i] + words_y[j])/2)) + " "
                        break
                if not paired[i]:
                    # 直接写入到字符中
                    row_prd += words_l[i]+ " " + str(words_x[i]) + " " + str(words_y[i]) + " "
            w_prd.writelines(row_prd.strip()+"\n")
            print ('page removed',imgName,rmcount)
        print("total dumplicate words is ", wordDump)
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data generation.')
    parser.add_argument('-i', '--input_file', default='../prd_valid_row.csv')
    parser.add_argument('-o', '--output_file', default='./prd_result.csv')
    args = parser.parse_args()
    removedumplict(args.input_file,args.output_file)
