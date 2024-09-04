#ifndef BUMPER_H
#define BUMPER_H
#include <string>
using namespace std;
namespace ly
{
#define MAX_LOSS_BUMP_CONUT 35
#define MAX_DETECT_BUMP_COUNT 5
    enum DETECT_MODE
    {
        NOT_GET_TARGET = 0,       // 未获得目标
        CONTINOUS_GET_TARGET = 1, // 连续获得目标
        LOST_BUMP = 2,            // 缓冲阶段
        DETECT_BUMP = 3           // 进入连续识别状态的缓冲
    };
    class Bumper
    {

    public:
        Bumper() = default;
        ~Bumper() = default;

        // 获取检测模式
        int getDetectMode(int &target_class, int &last_frame_class)
        {

            // 初始没有目标    ---检测到-->  检测缓冲状态


            // 当前无目标
            if (mode == NOT_GET_TARGET)   
            {
                if (target_class > 0)
                {
                    mode = DETECT_BUMP;         // 进入检测缓冲
                }
                else if (target_class <= 0)
                {
                    mode = NOT_GET_TARGET;      // 没有目标
                }
            }

            // 检测缓冲     --连续检测到--->    连续识别状态      
            // 当前处于检测缓冲中
            else if (mode == DETECT_BUMP)
            {
                // 检测缓冲++
                if (target_class == last_frame_class && target_class > 0)
                {
                    detect_bump_count++;
                }
                // 
                else if (target_class <= 0)
                {
                    detect_bump_count = 0;
                    mode = NOT_GET_TARGET;
                }

                if (detect_bump_count > MAX_DETECT_BUMP_COUNT)  // 连续检测到 5 次，进入连续识别
                {
                    mode = CONTINOUS_GET_TARGET;
                    detect_bump_count = 0;         
                }
            }

            // 没什么大毛病

            // 连续识别后面，才会进入这里
            // 掉帧多次，进入未检测到目标
            else if (mode == LOST_BUMP)
            {
                if (target_class <= 0)      // 丢失目标
                {
                    target_loss_count++;
                }
                else if (target_class > 0)
                {  
                    target_loss_count = 0;
                    mode = CONTINOUS_GET_TARGET;    // 连续识别状态
                }
                if (target_loss_count > MAX_LOSS_BUMP_CONUT)    // 连续识别不到次数过多，进入没有检测到
                {
                    mode = NOT_GET_TARGET;
                    target_loss_count = 0;
                }
            }
            // 连续识别状态     ---丢失帧数--->  掉帧缓冲状态

            // 连续识别状态
            else if (mode == CONTINOUS_GET_TARGET)
            {
                // 进入掉帧缓冲
                if (target_class <= 0 || target_class != last_frame_class)
                {
                    mode = LOST_BUMP;
                }
            }
            return mode;
        }

    private:
        int target_loss_count = 0;
        int detect_bump_count = 0;
        int mode = NOT_GET_TARGET;
    };
}
#endif