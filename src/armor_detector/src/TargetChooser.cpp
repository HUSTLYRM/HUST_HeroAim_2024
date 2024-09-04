#include "TargetChooser.h"
namespace ly
{
    TargetChooser::TargetChooser()
    {
    }

    TargetChooser::~TargetChooser()
    {
    }

    int TargetChooser::getPriority(int id, int last_id)
    {
        if (id == top_pri)
        {
            return 0;               // 最高优先级是操作手右击按下后，距离中心最近的那个类别
        }
        else if (id == last_id)       // 优先上一次选中的车
        {
            return 1;
        }
        else if (id == 1)       // 然后是英雄
        {
            return 2;
        }

        else if (id == 3 || id == 4 || id == 5) // 其次是步兵
        {
            return 3;
        }

        else if (id == 6)          // 然后是哨兵
        {
            return 4;
        }

        else if (id == 8)          // 最后基地 
        {
            return 5;
        }
        else                        // 其余，就是7 前哨战
        {
            return 6;
        }
    }

    // 按照优先级来的
    ArmorBlobs TargetChooser::getAimTarget(ArmorBlobs &blobs, int last_id)
    {
        // DLOG(ERROR) << "LAST ID " << last_id << endl;
        vector<int> target_candicate_indice;        // 候选index 
        ArmorBlobs shoot_target_candidate;          // 射击候选
        int now_priority = 6;                       // 初始最低优先级
        for (int i = 0; i < blobs.size(); i++)      // 遍历当前找到的所有车
        {
            if (getPriority(blobs[i]._class, last_id) < now_priority) // 有优先级更高的目标，清空，重新加载
            {
                now_priority = getPriority(blobs[i]._class, last_id); // 获取优先级
                shoot_target_candidate.clear();                         
                shoot_target_candidate.emplace_back(blobs[i]);          
                target_candicate_indice.emplace_back(i);
            }
            else if (getPriority(blobs[i]._class, last_id) == now_priority) // 有相同优先级的目标， 那么就讲所有优先级一样的装甲板加入进来
            {
                shoot_target_candidate.emplace_back(blobs[i]);        // 加入到射击目标中
                target_candicate_indice.emplace_back(i);              // 候选index
            }
        }

        // 获取选中装甲板id的一个索引
        target_indice = target_candicate_indice[0];
        
        // 将射击的目标进行距离排序
        getTargetOrder(shoot_target_candidate);

        // 返回armorblobs, 根据优先级识别到的候选 装甲板
        return shoot_target_candidate;
    }

    // 获取距离中心的距离
    float TargetChooser::getDistanceToCenter(const ArmorBlob &blob)
    {
        const cv::Point2f center_point = cv::Point2f(640, 512);
        cv::Point2f armor_blob_center = cv::Point2f(blob.rect.x + blob.rect.width / 2, blob.rect.y + blob.rect.height / 2.0);
        cv::Point2f dif = armor_blob_center - center_point;
        return dif.x * dif.x + dif.y * dif.y;
    }

    bool TargetChooser::compFunc(const ArmorBlob &blob1, const ArmorBlob &blob2)
    {
        return getDistanceToCenter(blob1) > getDistanceToCenter(blob2);
    }

    // 根据距离进行排序，优先选距离最近的
    void TargetChooser::getTargetOrder(ArmorBlobs &blobs)
    {
        sort(blobs.begin(), blobs.end(), compFunc);
    }
}