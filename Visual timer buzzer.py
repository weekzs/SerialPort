import random
import torch
import cv2
import numpy as np
import time
import onnxruntime as ort
import serial_scan
import myBeep
import winsound
from utils.general import non_max_suppression, scale_boxes
import threading
# 模型加载
model_pb_path = r"F:\垃圾分类\yolov5\runs\train\exp18\weights\best.onnx"

so = ort.SessionOptions()
net = ort.InferenceSession(model_pb_path, so)
def infer_img(img0, net, model_h, model_w, conf_thres=0.25, iou_thres=0.45):
    # 图像预处理
    img = cv2.resize(img0, (model_w, model_h))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # 模型推理
    outputs = net.run(None, {net.get_inputs()[0].name: img})[0]
    outputs = torch.from_numpy(outputs)

    # 应用NMS
    pred = non_max_suppression(outputs, conf_thres, iou_thres, agnostic=False)

    # 处理检测结果
    detections = []
    for det in pred:
        if det is not None and len(det):
            # 调整框尺寸到原始图像尺寸
            det[:, :4] = scale_boxes((model_h, model_w), det[:, :4], img0.shape).round()
            detections.append(det.numpy())

    if len(detections) > 0:
        detections = np.concatenate(detections)
        boxes = detections[:, :4]
        confs = detections[:, 4]
        ids = detections[:, 5].astype(int)
        return boxes, confs, ids
    else:
        return [], [], []


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # 坐标转换和绘制矩形框
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2 + 1)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2_text = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2_text, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def check_elements_in_list(list1, list2):
    # 如果 list1 为空，直接返回 False 和 'q'
    if len(list1) == 0:
        return False, 'q'

    # 遍历 list1 中的每个元素
    for item in list1:
        # 如果元素在 list2 中，返回 True 和该元素
        if item in list2:
            return True, item

    # 如果遍历完 list1 也没有找到匹配的元素，返回 False 和 'q'
    return False, 'q'
video = 1  # 1是外置摄像头
cap = cv2.VideoCapture(video)


def MYtest():
    non = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    dic_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    # 视频输入配置
    model_w, model_h = 640, 640  # 必须与训练时的图像尺寸一致
    conf_thres = 0.25            # 置信度阈值
    iou_thres = 0.45             # NMS阈值

    flag_det = True
    successd = 0

    while True:
        success, img0 = cap.read()
        if success and flag_det:
            t1 = time.time()
            boxes, confs, ids = infer_img(img0, net, model_h, model_w, conf_thres, iou_thres)
            t2 = time.time()

            if len(boxes) > 0:
                # 转换ids为字符串标签列表
                label = [dic_labels[id] for id in ids]
                successd, item = check_elements_in_list(label, non)

                # 绘制检测框
                for box, conf, id in zip(boxes, confs, ids):
                    label = f'{dic_labels[id]} {conf:.2f}'
                    plot_one_box(box, img0, label=label, color=(0, 255, 0))

                # 显示FPS
                fps = 1 / (t2 - t1)
                cv2.putText(img0, f'FPS: {fps:.2f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", img0)

        key = cv2.waitKey(1) & 0xFF
        if successd:
            return 1, item
        elif key == ord('q'):
            break
ser = serial_scan.MYscan()#这个改成全局变量
def running(it):
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
        while True:
            com_input = ser.read(1) #这里是交互用的
           # com_input = 1
            if com_input:  # 如果读取结果非空，则输出
                #myBeep.beep_sound()
                #time.sleep(1)

                print("开始识别{}".format(com_input))
                break
        non = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        non1 = ['1']
        non2 = ['2']
        non3 = ['3']
        non4 = ['4']
        non5 = ['5']
        non6 = ['6']
        non7 = ['7']
        non8 = ['8']
        non9 = ['9']
        non0 = ['0']

        try:
            # while True:
            #识别的在it里面直接发字符‘1’
            if it in non:
                serial_scan.send(ser, '1')
                #这个是识别成功之后，蜂鸣器响
                myBeep.beep_sound()#这个顺序或许可以放在接收到单边机的信号之后？
                time.sleep(1)

            #serial_scan.receive(ser,1)						# 前面两行可以注释，换成后面这个函数
        except KeyboardInterrupt:
            if ser != None:
                print("close serial port")
                ser.close()
        #write_len = ser.write("".encode('utf-8'))
#下面是定时中断线程e
def exit_program():
    #终端测试
    print("Time's up! Exiting...")
    #发回去的数据
    #serial_scan.send(ser, '7')
#全局变量

mytime = 10#这里是秒数

if __name__ == "__main__":
    timer = threading.Timer(mytime, exit_program)
    timer.start()
    while True:
        result, it = MYtest()
        running(it)
