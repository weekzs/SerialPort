import cv2
import numpy as np
import time
import onnxruntime as ort
import torch
import random
from utils.general import non_max_suppression, scale_boxes


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


if __name__ == "__main__":
    # 模型加载
    model_pb_path = "D:/桌面/代码文件/垃圾分类/yolov5/runs/train/exp7/weights/best.onnx"
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)

    # 标签字典
    dic_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    # 视频输入配置
    cap = cv2.VideoCapture(1)
    model_w, model_h = 640, 640  # 必须与训练时的图像尺寸一致

    while True:
        success, img0 = cap.read()
        if success:
            t1 = time.time()
            boxes, confs, ids = infer_img(img0, net, model_h, model_w)
            t2 = time.time()

            if len(boxes) > 0:
                for box, conf, id in zip(boxes, confs, ids):
                    label = f'{dic_labels[id]} {conf:.2f}'
                    plot_one_box(box, img0, label=label, color=(0, 255, 0))

                fps = 1 / (t2 - t1)
                cv2.putText(img0, f'FPS: {fps:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Detection", img0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
