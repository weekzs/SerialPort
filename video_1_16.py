import random

import cv2
import numpy as np
import time
import onnxruntime as ort
import serial_scan

# 模型加载
model_pb_path = r"D:\桌面\代码文件\垃圾分类\yolov5\runs\train\exp7\weights\best.onnx"
so = ort.SessionOptions()
net = ort.InferenceSession(model_pb_path, so)
def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs
def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)

    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
        描述： 在图像 img 上绘制一个边界框，
                 这个函数来自YoLov5项目。
    参数：
        x：一个盒子喜欢 [x1，y1，x2，y2]
        img：OpenCV 映像对象
        color：绘制矩形的颜色，如（0,255,0）
        标签： str
        line_thickness：int
    返回：
        不归路
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, [model_w, model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)

    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

    return boxes, confs, ids


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

    # 标签字典
    non = ['0','1','2','3','4','5','6','7','8','9']
    dic_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}  # 修改为自己的类别

    # 模型参数
    model_h = 640  # 图片resize的大小
    model_w = 640
    nl = 3  # 三层输出对应类别
    na = 3  # 每层3种anchor
    stride = [8., 16., 32.]  # 缩放尺度因子
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]  # 默认anchors大小设置
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    # video = 1  # 1是外置摄像头
    # cap = cv2.VideoCapture(video)

    flag_det = True  # 控制检测开关
    successd = 0 #全局变量，识别是否成功，在列表里面
    while True:
        success, img0 = cap.read()
        if success:
            if flag_det:
                t1 = time.time()
                det_boxes, scores, ids = infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid,
                                                   thred_nms=0.4, thred_cond=0.5)
                # print(det_boxes)
                t2 = time.time()
                if det_boxes is not None and scores is not None and ids is not None:
                    label = [dic_labels[i] for i in ids]
                    successd, item = check_elements_in_list(label, non)
                    for box, score, id in zip(det_boxes, scores, ids):
                        # successd, item = check_elements_in_list(id, non)
                        label = '%s:%.2f' % (dic_labels[id], score)
                        plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)
                        print(f"识别到的类别：{dic_labels[id]}")  # 打印识别到的类别
                        # print(f"Box coordinates: {box}")
                        str_FPS = "FPS: %.2f" % (1. / (t2 - t1))
                        plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=3)

                        cv2.putText(img0, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("video", img0)

        key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):  # 按q退出
        #     break
        # elif key & 0xFF == ord('s'):  # 按s切换检测开关
        #     flag_det = not flag_det
        #     print(flag_det)
        if successd:
            # cap.release()
            # cv2.destroyAllWindows()
            return 1, item
def running(it):
    ser = serial_scan.MYscan()
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
        while True:
           # com_input = ser.read(1) 这里是交互用的
            com_input = 1
            if com_input:  # 如果读取结果非空，则输出
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
            if it in non1:
                serial_scan.send(ser, '1')
            if it in non2:
                serial_scan.send(ser, '2')
            if it in non3:
                serial_scan.send(ser, '3')
            if it in non4:
                serial_scan.send(ser, '4')
            if it in non5:
                serial_scan.send(ser, '5')
            if it in non6:
                serial_scan.send(ser, '6')
            if it in non7:
                serial_scan.send(ser, '7')
            if it in non8:
                serial_scan.send(ser, '8')
            if it in non9:
                serial_scan.send(ser, '9')
            if it in non0:
                serial_scan.send(ser, '0')
            serial_scan.receive(ser,1)						# 前面两行可以注释，换成后面这个函数
        except KeyboardInterrupt:
            if ser != None:
                print("close serial port")
                ser.close()
        #write_len = ser.write("".encode('utf-8'))
if __name__ == "__main__":
    while True:
          result, it = MYtest()
          running(it)