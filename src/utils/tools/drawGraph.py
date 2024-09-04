#-*- coding : utf-8-*-
import socket
import matplotlib.pyplot as plt
import numpy as np

HOST = '127.0.0.1'
PORT = 9999
NUM = 20

class Server():
    def __init__(self, ip, port):
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.bind((ip, port))
        self.socket_server.listen(5)

if __name__ == '__main__':

    server = Server(HOST, PORT)
    client_socket, address = server.socket_server.accept()
    axis_x = []
    queue_x = []
    queue_z = []
    queue_xp = []
    queue_zp = []
    cnt = 0
    skip = 0
    # plt.ion()
    fig, ax = plt.subplots()
    
    while True:

        recvmsg = client_socket.recv(40)
        # data = str(recvmsg)
        data = recvmsg.decode("utf-8", "ignore")
        # print("recv: {}".format(data))

        skip += 1
        if skip < 50:
            continue
        skip = 0
        print(data)
        speed = data.split(' ');
        x = float(speed[0])
        z = float(speed[1])
        x_p = float(speed[2])
        z_p = float(speed[3])
        print("x: {}, z: {}".format(x, z))
        
        axis_x.append(cnt)
        queue_x.append(x)
        queue_z.append(z)
        queue_xp.append(x_p)
        queue_zp.append(z_p)
        
        cnt += 1

        if len(axis_x) > 40:
            axis_x = axis_x[20:]
            queue_x = queue_x[20:]
            queue_xp = queue_xp[20:]
            queue_z = queue_z[20:]
            queue_zp = queue_zp[20:]
        
        plt.xlim(1, NUM)
        plt.ylim(-1, 1)
        ax.cla()
        ax.plot(axis_x, queue_x, 'r', lw=1) # draw line chart
        ax.plot(axis_x, queue_xp, 'b', lw=1) # draw line chart
        
        # ax.plot(axis_x, queue_z, 'r', lw=1) # draw line chart
        # ax.plot(axis_x, queue_zp, 'b', lw=1) # draw line chart
        plt.pause(0.005)
    # plt.ioff()
    plt.show()
    server.socket_server.close()


# import matplotlib.pyplot as plt
# from random import random
 
# def do_something(res):
#     for p in range(10000000):
#         res += p
 
# fig, ax = plt.subplots()
# x = []
# y = []
# res = 0
# for i in range(50):
#     x.append(i)
#     y.append(50*random())
#     ax.cla() # clear plot
#     ax.plot(x, y, 'r', lw=1) # draw line chart
#     # ax.bar(y, height=y, width=0.3) # draw bar chart
#     do_something(res)    
#     plt.pause(0.1)