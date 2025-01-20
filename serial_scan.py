import serial.tools.list_ports
def MYscan():
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("无串口设备。")
        return None
    else:
        print("可用的串口设备如下：")
        for comport in ports_list:
            print(list(comport)[0], list(comport)[1])
        # 这里假设我们使用第一个找到的串口
        return serial.Serial(list(comport)[0], 115200)
def send(ser, send_data):
    if (ser.isOpen()):
        ser.write(send_data.encode('utf-8'))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")
def receive(ser, num_bytes=1):  # num_bytes 表示要接收的字节数，默认为 1
    if ser.isOpen():
        received_data = ser.read(num_bytes)  # 从串口读取 num_bytes 个字节的数据
        print("接收成功", received_data)
        return received_data
    else:
        print("接收失败！")
        return None
if __name__ == "__main__":
    ser = MYscan()
    #ser = serial.Serial("COM8", 115200)  # 打开/dev/ttyAMA10，将波特率配置为115200，其余参数使用默认值
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
        write_len = ser.write("".encode('utf-8'))
        send(ser,'1')
        receive(ser,1)