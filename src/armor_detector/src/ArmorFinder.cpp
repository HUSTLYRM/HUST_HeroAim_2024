//
// Created by zhiyu on 2021/8/24.
//

#include "ArmorFinder.h"

using namespace ly;

bool ArmorFinder::matchTwoLightBar(const RotatedRect &l, const RotatedRect &r) {
    status = checkAngleDiff(l, r);
    // DLOG_IF(INFO, !status) << "angle differs: left: " << getAngle(l) << " right: " << getAngle(r);
    if(!status) return false;
    status = checkHeightDiff(l, r);
    // DLOG_IF(INFO, !status) << "height differs: " << abs((l.center-r.center).y);
    if(!status) return false;
    // status = checkRatio(l, r);
    // if(!status) return false;
    status = checkHeightMatch(l, r);
    // DLOG_IF(INFO, status) << "height matches: " << l.size.height << " " << r.size.height;
    if(!status) return false;
    return true;
}

bool ArmorFinder::judgeArmor(const ArmorBlob &armor_blob) {
    return armor_blob.rect.size().aspectRatio() <= 5 && armor_blob.rect.size().aspectRatio() > 1.5;
}

/***
 * @brief 将两个灯条拼接成一个装甲
 * @param l
 * @param r
 * @param armor
 * @return
 */
bool ArmorFinder::getArmor(const RotatedRect &l, const RotatedRect &r, ArmorBlob& armor) {
    Point2f points_of_rrect[4];
    l.points(points_of_rrect);
    float height = fmax(l.size.width, l.size.height);
    armor.rect = Rect(l.center.x, l.center.y-height/2, r.center.x-l.center.x, height);

    // armor
    // 0 1
    // 3 2
//    cout << l.angle << " " << r.angle << endl;

    // if(l.angle > 45){
    //     armor.corners[0] = points_of_rrect[0];
    //     armor.corners[3] = points_of_rrect[3];
    // } else{
    //     armor.corners[0] = points_of_rrect[1];
    //     armor.corners[3] = points_of_rrect[0];
    // }
    // r.points(points_of_rrect);
    // if(r.angle > 45){
    //     armor.corners[1] = points_of_rrect[1];
    //     armor.corners[2] = points_of_rrect[2];
    // } else{
    //     armor.corners[1] = points_of_rrect[2];
    //     armor.corners[2] = points_of_rrect[3];
    // }

    // if(l.angle > 45){
    //     armor.corners[0] = points_of_rrect[1];
    //     armor.corners[3] = points_of_rrect[2];
    // } else{
    //     armor.corners[0] = points_of_rrect[2];
    //     armor.corners[3] = points_of_rrect[3];
    // }
    // r.points(points_of_rrect);
    // if(r.angle > 45){
    //     armor.corners[1] = points_of_rrect[0];
    //     armor.corners[2] = points_of_rrect[3];
    // } else{
    //     armor.corners[1] = points_of_rrect[1];
    //     armor.corners[2] = points_of_rrect[0];
    // }
    
    // 使用的是resnet18，但是1x32x32的输入，将两个点重新选回了灯条的中心，影响不大（之前说是增加BA优化后, 这样可以让yaw轴解算更加准确）
    // TODO 修改代码中的装甲板的四点的选取策略，使用half_length 0.675f进行计算
    // 选取装甲板的中心进行后续求解，这样可以减少装甲板yaw值的误差，同时提供pnp解算的准确度
    if (l.angle > 45)
    {
        armor.corners[0] = (points_of_rrect[0] + points_of_rrect[1]) / 2;
        armor.corners[3] = (points_of_rrect[3] + points_of_rrect[2]) / 2;
    }
    else
    {
        armor.corners[0] = (points_of_rrect[1] + points_of_rrect[2]) / 2;
        armor.corners[3] = (points_of_rrect[0] + points_of_rrect[3]) / 2;
    }
    r.points(points_of_rrect);
    if (r.angle > 45)
    {
        armor.corners[1] = (points_of_rrect[1] + points_of_rrect[0]) / 2;
        armor.corners[2] = (points_of_rrect[2] + points_of_rrect[3]) / 2;
    }
    else
    {
        armor.corners[1] = (points_of_rrect[2] + points_of_rrect[1]) / 2;
        armor.corners[2] = (points_of_rrect[3] + points_of_rrect[0]) / 2;
    }

    if(max(armor.corners[0].x, armor.corners[3].x) > min(armor.corners[1].x, armor.corners[2].x)) return false;
    return true;
}

// 检查两个灯条角度是否合理
bool ArmorFinder::checkAngleDiff(const RotatedRect &l, const RotatedRect &r) {
    const float & angle_l = getAngle(l);
    const float & angle_r = getAngle(r);
    // 灯条角度差
    return abs(angle_l-angle_r) < 30;
}

// 获取灯条角度，RotatedRect 角度为 -90到90，角度为负数时表示向右倾斜，为正数时表示向左倾斜
float ArmorFinder::getAngle(const RotatedRect &rrect) {
    // 宽 > 高
    return rrect.size.width > rrect.size.height ? rrect.angle-90 : rrect.angle;
}

// 检查中心距离差 是否大致符合
bool ArmorFinder::checkHeightDiff(const RotatedRect &l, const RotatedRect &r) {
    const Point2f& diff = l.center - r.center;
    return abs(diff.y) < min(max(l.size.height, l.size.width), max(r.size.height, r.size.width))*1.5;
}

bool ArmorFinder::checkHorizontalDistance(const RotatedRect &l, const RotatedRect &r) {
    return false;
}

bool ArmorFinder::checkDislocation(const RotatedRect &l, const RotatedRect &r) {
    return false;
}

// Rect ArmorFinder::getScaleArmorToRoi(const Rect& rect) {
//     // w*5/4 h*2
//     int x = max(0, rect.x-rect.width/8);
//     int y = max(0, rect.y-rect.height/2);
//     int width = min(x + rect.width*5/4, 1280) - x;
//     int height = min(y + rect.height*2, 1024) - y;
//     return {x, y, width, height};
// }

// 获取装甲板的极限边界坐标
vector<int> ArmorFinder::getExtreme(const ArmorBlob& armor) {
    
    int x1 = min(1280.f, max(0.f, min(armor.corners[0].x, armor.corners[3].x)));
    int y1 = min(1280.f, max(0.f, min(armor.corners[0].y, armor.corners[1].y) - min(armor.rect.height, armor.rect.width)/2.5f));
    int x2 = min(1280.f, max(0.f, max(armor.corners[2].x, armor.corners[1].x)));
    int y2 = min(1024.f, max(0.f, max(armor.corners[2].y, armor.corners[3].y) + min(armor.rect.height, armor.rect.width)/2.5f));
    
    return {x1, y1, x2, y2};
}

// 检查左右灯条的长度不是差别太大
bool ArmorFinder::checkHeightMatch(const RotatedRect &l, const RotatedRect &r) {
    float lh = max(l.size.height, l.size.width);
    float rh = max(r.size.height, r.size.width);
    return min(lh, rh) * 2 > max(lh, rh);
}


