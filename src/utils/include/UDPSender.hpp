#ifndef UDP_SENDER_HPP
#define UDP_SENDER_HPP

#include <string>
#include <boost/asio.hpp>

using boost::asio::ip::udp;

namespace ly
{   
    /**
     * @brief 位姿解算数据帧，tail是VOFA+调试助手规定的帧尾
     * 并且选用了justFloat模式： 发送的数据都是浮点数
     * 这部分需要补充，但是必须使用相同的tail
     * 参考这部分补充内容
     */
    struct PoseDataFrame
    {
        float x;
        float y;
        float z;
        float pitch;
        float yaw;
        float roll;
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };

    struct AimFrame
    {
        // 当前值
        float x_now;
        float y_now;
        float z_now;

        // 预测值
        float x_pre;
        float y_pre;
        float z_pre;
        
        // 正常时间
        float time_raw;
        // 预测时间
        float time_pre;
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };

    struct VehicleFrame
    {
        float x_c;
        float y_c;
        float z_c;
        float yaw;
        float v_x;
        float v_y;
        float v_z;
        float v_yaw;
        float r;
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };

    struct BuffFrame
    {
        float x_now;
        float y_now;
        float x_pre;
        float y_pre;
        float time_raw;
        float time_pre;
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };

    struct DeltatFrame
    {
        float delta_t;
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };

    // 前哨站中心的xyz
    struct CenterFrame
    {
        float x;        // 中心的x
        float y;        // 中心的y
        float z;        // 中心的z
        float pitch;    // 当前 pitch
        float yaw;      // 当前 yaw
        float distance; // 距离前哨站的距离
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };

    // 辅瞄模式调试卡尔曼滤波使用的帧
    struct AimKFFrame
    {
        float xn; float pred_xn;
        float zn; float pred_zn;
        float xv; float pred_xv;
        float zv; float pred_zv;
        // 主要观测 position 和 speed
        float xa; float pred_xa;
        float za; float pred_za;
        float reset;
        unsigned char tail[4]{0x00, 0x00, 0x80, 0x7f};
    };


    class UDPSender
    {
    public:
        /**
         * @brief Construct a new UDPSender object
         * 
         * @param ip_address 目标ip地址，在这里指定为192.168.1.3, 也就是你的windows的ip地址
         * @param port 目标端口，同样为windows的端口，在VOFA+中使用的部分，可以方便的
         */
        UDPSender(const std::string &ip_address, int port)
            : remote_endpoint_(boost::asio::ip::address::from_string(ip_address), port),
              socket_(io_service_, udp::endpoint(udp::v4(), 0))
        {
        }

        /**
         * @brief 发送数据，这里建议
         * 
         * @tparam T 
         * @param data 
         */
        template <typename T>
        void send(const T &data)
        {
            std::string message(sizeof(data), '\0');
            std::memcpy(message.data(), &data, sizeof(data));

            socket_.send_to(boost::asio::buffer(message), remote_endpoint_);
        }

    private:
        boost::asio::io_service io_service_;
        udp::endpoint remote_endpoint_;
        udp::socket socket_;
    };
}

#endif // UDP_SENDER_HPP