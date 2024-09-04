//
// Created by zhiyu on 2022/7/30.
//

#ifndef AUTOAIM_ARRAY_H
#define AUTOAIM_ARRAY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

namespace ly{
    template <typename T>
    class RollingArray                          // 循环数组
    {
    public:
        RollingArray():
            _size(100), id(0), metric(0){}
        
        RollingArray(int _size):
            _size(_size), id(0), metric(0){}

        void update(T elem){
            // 3D点
            if(array.size() < _size){       // 加入到数据中        
                metric += elem;             // metric 加上装甲板的3d点
                array.push_back(elem);      // 加入到队列中
                return;
            }
            metric -= array[id];            // 减去
            array[id] = elem;
            metric += array[id];
            id = ((id + 1) % _size + _size) % _size;    // id循环
        }

        T getMetric(){ return metric; }
        // 读取
        int size(){ return array.size(); }
        // 读取当前的数据
        int getSize(){ return _size; }
        void clear(){ 
            array.clear();
            metric = {};
        }
    private:
        int id;
        int _size;
        vector<T> array;
        T metric;
    };
}

#endif