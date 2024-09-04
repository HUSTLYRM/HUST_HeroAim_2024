//
// Created by zhiyu on 2022/7/30.
//

#ifndef AUTOAIM_NEWARRAY_H
#define AUTOAIM_NEWARRAY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

namespace ly
{
    class newIntRollingArray // 循环数组
    {
    public:
        newIntRollingArray() : _size(100), id(0) {}

        newIntRollingArray(int _size) : _size(_size), id(0) {}

        void updat(double elem)
        {
            // 3D点
            if (array.size() < _size)
            {                          // 加入到数据中
                array.push_back(elem); // 加入到队列中
                return;
            }
            id = ((id + 1) % _size + _size) % _size; // id循环
        }

        double getMinimumPoint(){
            double min = array[0];
            for(int i=1;i<array.size();i++){
                if(min > array[i]){
                    min = array[i];
                }
            }
            return min;
        }
        
        double getMidPoint(){
            double min = array[0];
            double max = array[0];
            for(int i=1;i<array.size();i++){
                if( min > array[i]){
                    min = array[i];
                }
                if( max < array[i]){
                    min = array[i];
                }
            }
            return (min+max)/2;
        }

        int size() { return array.size(); }
        
        void clear()
        {
            array.clear();
        }

    private:
        int id;
        int _size;
        vector<double> array;
    };

    class newRollingArray // 循环数组
    {
    public:
        newRollingArray() : _size(100), id(0), metric(0) {}

        newRollingArray(int _size) : _size(_size), id(0), metric(0) {}

        void update(cv::Point3d elem)
        {
            // 3D点
            if (array.size() < _size)
            {                          // 加入到数据中
                metric += elem;        // metric 加上装甲板的3d点
                array.push_back(elem); // 加入到队列中
                return;
            }
            metric -= array[id]; // 减去
            array[id] = elem;
            metric += array[id];
            id = ((id + 1) % _size + _size) % _size; // id循环
        }

        cv::Point3d getMetric() { return metric; }
        // 读取
        int size() { return array.size(); }
        // 读取当前的数据
        int getSize() { return _size; }
        void clear()
        {
            array.clear();
            metric = {};
        }
        
        // cv::Point3d computeSquaredDifference(const cv::Point3d &a, const cv::Point3d &b)
        // {
        //     cv::Point3d difference(a.x - b.x, a.y - b.y, a.z - b.z);
        //     cv::Point3d squaredDifference(difference.x * difference.x, difference.y * difference.y, difference.z * difference.z);
        //     return squaredDifference;
        // }

        // double computeOutlierThreshold(const std::vector<cv::Point3d> &points)
        // {
        //     cv::Point3d meanPoint(0, 0, 0);
        //     for (const auto &point : points)
        //     {
        //         meanPoint += point;
        //     }

        //     meanPoint /= static_cast<double>(points.size());

        //     cv::Point3d squaredDifferenceSum(0, 0, 0);
        //     for (const auto &point : points)
        //     {
        //         cv::Point3d difference = computeSquaredDifference(point, meanPoint);
        //         squaredDifferenceSum += difference;
        //     }

        //     cv::Point3d variance = squaredDifferenceSum / static_cast<double>(points.size());
        //     cv::Point3d standardDeviation(std::sqrt(variance.x), std::sqrt(variance.y), std::sqrt(variance.z));

        //     double threshold = 2.0 * norm(standardDeviation);
        //     return threshold;
        // }

        // double calculateDistanceToOrigin(const cv::Point3d &point)
        // {
        //     double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        //     return distance;
        // }


        // 获取距离我最近的numPoints个点
        std::vector<cv::Point3d> getNearestPoints(int numPoints)
        {
            // 注意用深拷贝
            std::vector<cv::Point3d> sortedArray(array.begin(), array.end());
            // std::vector<cv::Point3d> sortedArray = array;
            std::sort(sortedArray.begin(), sortedArray.end(), [](const cv::Point3d &a, const cv::Point3d &b)
                      {
                float distanceA = cv::norm(a);
                float distanceB = cv::norm(b);
                return distanceA < distanceB; });

            // 防止越界
            int numPointsToConsider = std::min(numPoints, static_cast<int>(sortedArray.size()));

            // 获取最近的一些
            std::vector<cv::Point3d> nearestPoints(sortedArray.begin(), sortedArray.begin() + numPointsToConsider);

            return nearestPoints;
        }

        cv::Point3d getMinimumPoint(){
            cv::Point3d min = array[0];
            for(int i=1;i<array.size();i++){
                if(cv::norm(min) > cv::norm(array[i])){
                    min = array[i];
                }
            }
            return min;
        }

        // 对最近的一些点进行均值
        cv::Point3d getMeanOfNearestPoints(int numPoints, int numFilteredPoints = 0)
        {
            std::vector<cv::Point3d> nearestPoints = getNearestPoints(numPoints);   // 从小到大排序的

            // 确定离群点阈值范围
            // double threshold = computeOutlierThreshold(nearestPoints);

            // 过滤离群点
            // std::vector<cv::Point3d> filteredPoints;
            // for (const auto &point : nearestPoints)
            // {
            //     double distanceToOrigin = calculateDistanceToOrigin(point);
            //     if (distanceToOrigin <= threshold)
            //     {
            //         filteredPoints.push_back(point);
            //     }
            // }
            cv::Point3d meanPoint(0, 0, 0);

            // 剔除其中，距离我最近的那些点
            for (int i = numFilteredPoints;i < numPoints;i++)
            {
                meanPoint += nearestPoints[i];
            }
            meanPoint /= (numPoints - numFilteredPoints);
            return meanPoint;
        }

    private:
        int id;
        int _size;
        vector<cv::Point3d> array;
        cv::Point3d metric;
    };
}

#endif