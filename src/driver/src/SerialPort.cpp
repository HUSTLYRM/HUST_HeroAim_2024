//
// Created by zhiyu on 2021/8/20.
//

#include "../include/SerialPort.h"

namespace ly
{
    // 固定的从串口读取数据
    void SerialPort::read_data(Params_ToSerialPort& params)
    {
        while (1)
        {
            try
            {
                this->readData(params.data);
            }
            catch (...)
            {
                DLOG(ERROR) << "catch an error in SerialPort::read_data";
                break;
            }
            usleep(1000);                   // 规定的发送频率
        }
        return;
    }

    SerialPort::SerialPort()
    {
        new (this) SerialPort("/dev/ttyACM0");
    }

    SerialPort::SerialPort(const string &port_name)
    {
        try
        {
            LOG(INFO)<<"Try to connect to: "<<port_name<<endl;
            // 创建串口对象
            this->_serial_port = new boost::asio::serial_port(_io_service, port_name);
            // 设置波特率
//            this->_serial_port->set_option(boost::asio::serial_port::baud_rate(115200));
            this->_serial_port->set_option(boost::asio::serial_port::baud_rate(1000000));

            // 流量控制
            this->_serial_port->set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
            // 奇偶校验
            this->_serial_port->set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
            // 1位停止位
            this->_serial_port->set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
            // 8位数据位
            this->_serial_port->set_option(boost::asio::serial_port::character_size(8));
            // 串口发送数据tmp
            _data_tmp = (uint8_t *)malloc((size_t)_data_len);
            pingpong = (uint8_t *)malloc((size_t)_data_len * 2);
        }
        catch (...)
        {
            LOG(ERROR) << "create serial port object error! ";
            string port_name_back = "/dev/ttyACM1";
            try {
                LOG(INFO)<<"Try to connect to: "<<port_name_back<<endl;
                // 创建串口对象
                this->_serial_port = new boost::asio::serial_port(_io_service, port_name_back);
                // 设置波特率
                this->_serial_port->set_option(boost::asio::serial_port::baud_rate(1000000));
                // 流量控制
                this->_serial_port->set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
                // 奇偶校验
                this->_serial_port->set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
                // 1位停止位
                this->_serial_port->set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
                // 8位数据位
                this->_serial_port->set_option(boost::asio::serial_port::character_size(8));
                // 串口发送数据tmp
                _data_tmp = (uint8_t *)malloc((size_t)_data_len);
                pingpong = (uint8_t *)malloc((size_t)_data_len * 2);
            }
            catch (...)
            {
                LOG(ERROR) << "create serial port object error! ";
            }
        }
    }

    SerialPort::~SerialPort()
    {
        free(_data_tmp);
        free(pingpong);
        delete _serial_port;
    }

    // boost底层从串口读取数据
    void SerialPort::serialPortRead(uint8_t *msg, uint8_t max_len)
    {
        try
        {
            read(*_serial_port, boost::asio::buffer(msg, max_len), _err);
        }
        catch (...)
        {
            LOG(ERROR) << "readData from serial port error! " << _err.message();
        }
    }

    // boost底层向串口写入数据
    void SerialPort::serialPortWrite(uint8_t *msg, int len)
    {
        try
        {
            write(*_serial_port, boost::asio::buffer(msg, (size_t)len));
        }
        catch (...)
        {
            LOG(ERROR) << "write to serial port error! ";
        }
    }

    void SerialPort::addCRC(SerialPortData *msg)
    {
        if (!msg)
            return;
        msg->crc = getCRC(msg);
    }

    void SerialPort::addCRC(unsigned char *msg)
    {
        if (!msg)
            return;
        msg[_data_len_write - 1] = getCRC(msg);
    }

    uint8_t SerialPort::getCRC(SerialPortData *data)
    {
        auto _data = reinterpret_cast<unsigned char *>(data);
        int dwLength = _data_len - 1;
        unsigned char ucCRC8 = CRC8_INIT;
        unsigned char ucIndex;
        while (dwLength--)
        {
            ucIndex = ucCRC8 ^ (*_data++);
            ucCRC8 = CRC8_TAB[ucIndex];
        }
        return ucCRC8;
    }

    uint8_t SerialPort::getCRC(unsigned char *data)
    {
        auto _data = reinterpret_cast<unsigned char *>(data);
        int dwLength = _data_len_write - 1;
        unsigned char ucCRC8 = CRC8_INIT;
        unsigned char ucIndex;
        while (dwLength--)
        {
            ucIndex = ucCRC8 ^ (*_data++);
            ucCRC8 = CRC8_TAB[ucIndex];
        }

        return ucCRC8;
    }

    bool SerialPort::verifyCRC(SerialPortData *data)
    {
        if (!data)
            return false;
        return data->crc == getCRC(data);
    }

    // 从串口读取imu数据
    void SerialPort::readData(SerialPortData *imu_data)
    {
        serialPortRead(_data_tmp, _data_len);
        memcpy(pingpong + _data_len, _data_tmp, _data_len);
        for (int start_bit = 0; start_bit < _data_len; start_bit++)
        {
            if (pingpong[start_bit] == '!')                                   // 起始位
            {
                memcpy(_data_tmp, pingpong + start_bit, _data_len);
                if (verifyCRC(reinterpret_cast<SerialPortData *>(_data_tmp))) // 满足CRC校验
                {
                    imu_data->flag = _data_tmp[1];
                    imu_data->pitch = (_data_tmp[2] << 8)                     // pitch数据
                                      | _data_tmp[3];
                    imu_data->yaw = (((int)_data_tmp[4]) << 24)               // yaw轴数据
                                    | (((int)_data_tmp[5]) << 16)
                                    | (((int)_data_tmp[6]) << 8)
                                    | (_data_tmp[7]);
                    imu_data->color = _data_tmp[8];                           // 己方颜色
                    // 时间戳 是从 下位机读上来的
                    imu_data->time_stamp = (((int)_data_tmp[9]) << 24)        // 时间戳
                                           | (((int)_data_tmp[10]) << 16)
                                           | (((int)_data_tmp[11]) << 8)
                                           | (_data_tmp[12]);
                    imu_data->roll = (_data_tmp[13] << 8)                     // 己方roll轴
                                     | _data_tmp[14];
                    imu_data->right_clicked = _data_tmp[15];                  // 操作手右击（准备就绪）

                    // 读取
                    imu_data->user_time_bias = _data_tmp[16];                    // 射击速度

                    imu_data->outpost_state = _data_tmp[17];                     // 前哨状态
                    // DLOG(WARNING) << "color:" << to_string(imu_data->color)  << std::endl;
                    // DLOG(INFO) << "size of recv: " << _data_len << " right: " << imu_data->right_clicked << " " << _data_tmp[15] << " end";
                    // DLOG(INFO) << "readData: yaw " << imu_data->yaw << " pitch " << imu_data->pitch;
                    break;
                }
            }
        }
        memcpy(pingpong, pingpong + _data_len, _data_len);
    }

    // 向串口写入数据
    void SerialPort::writeData(SerialPortWriteData *_data_write)
    {
        msg[0] = '!';           // 起始位
        unsigned char *tmp;
        msg[1] = 0x05;
        tmp = (unsigned char *)(&_data_write->pitch);   // pitch轴数据， 2个字节
        msg[2] = tmp[1];
        msg[3] = tmp[0];
        tmp = (unsigned char *)(&_data_write->yaw);     // yaw轴数据， 4个字节
        msg[4] = tmp[3];
        msg[5] = tmp[2];
        msg[6] = tmp[1];
        msg[7] = tmp[0];

        msg[8] = _data_write->shootStatus;              // 涉及状态

        tmp = (unsigned char *)(&_data_write->time_stamp);  // 时间戳
        msg[9] = tmp[3];
        msg[10] = tmp[2];
        msg[11] = tmp[1];
        msg[12] = tmp[0];

        msg[13] = _data_write->state;                   // 当前状态
        msg[14] = _data_write->num;                     // 瞄准的id

        addCRC(msg);
        serialPortWrite(msg, _data_len_write);
    }
}