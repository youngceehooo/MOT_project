# MOT_project
the repository to be submited as our summer camp project 
## 小组成员：
杨子赫-U202112240\
杨呈恺-U202112236\
詹智杰-U202112243\
吴炳旭-U202112233\
## 项目关键技术
### 环境搭建（注意版本兼容）
1.torch(2.0.1+cu117,2.0.1+cu118)\
2.cuda(11.7,11.8)\
3.super_gradients(3.2.0)\
4.visual studio编译c++
### 导入的库
1.numpy（import numpy as np）：用于处理数值计算和数组操作的Python库。\
2.datetime（import datetime）：用于处理日期和时间的Python库。\
3.cv2（import cv2）：OpenCV库，用于计算机视觉任务，如图像处理和计算机视觉应用。\
4.torch（import torch）：PyTorch深度学习框架，用于构建和训练神经网络。\
5.absl（from absl import app, flags, logging）：Google的abseil库，用于处理命令行参数和日志记录。\
6.DeepSort（from deep_sort_realtime.deepsort_tracker import DeepSort）：一个用于多对象跟踪的深度学习模型。\
7.models（from super_gradients.training import models）：包含模型定义的模块。\
8.Models（from super_gradients.common.object_names import Models）：包含模型名称的模块。\
9.threading（import threading）：Python标准库的一部分，用于多线程编程。\
10.Queue（from queue import Queue）：Python标准库的一部分，用于创建队列数据结构。\
11.sys（import sys）：Python标准库的一部分，用于与Python解释器进行交互和访问系统相关信息
### 两个模型
___1.Deep_sort模型，创建deepsort跟踪器___\
___2.yolo_nas_1模型,用于目标检测___

## 项目功能代码：
**_本项目用于多目标追踪场景，提供目标选择、计算目标速度、从视频帧中截取目标图像、将选中目标单独监控等功能_**\
___以下代码均省略初始化参数___\
### 目标选择功能：
___创建鼠标回测函数___
   
    def on_mouse_click(event, x, y, flags, param):        
        nonlocal selected_object_id
        if event == cv2.EVENT_LBUTTONDOWN:  # 当鼠标左键点击时
            for idx, track in enumerate(tracks):
                ltrb = track.to_tlbr()  # 左上角和右下角坐标
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_object_id = idx
                    print(f"Selected object with ID: {selected_object_id}")
                    break'''
___鼠标回应函数___
    
    cv2.setMouseCallback("Frame", on_mouse_click)  # 设置鼠标事件回调函数
___遍历跟踪器进行处理___
    
    for idx, track in enumerate(tracks):
            if selected_object_id != -1 and idx != selected_object_id:
                continue
            if selected_object_id != -1 and idx == selected_object_id:
                # 在这里添加处理特定对象的代码
                ltrb = track.to_ltrb()  # 获取跟踪信息
                class_id = track.get_det_class()
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                captured_image = frame[y1:y2, x1:x2]  # 截取图像
                cv2.imwrite("captured_image.jpg", captured_image)
                # 删除选定对象的原有边界框和文本
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 用白色绘制边界框
                cv2.putText(frame, "", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 清空文本
        # 目标对象框设置为红色
                B, G, R = 0,0,255
        # 创建显示在帧上的文本
                class_name = class_names[class_id]
                text = f"{track_id} - {class_name}(Selected) "
        # 在帧上绘制边界框和文本
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
### 计算目标速度
 
    # 如果有之前的帧，计算速度
            if hasattr(track, 'has_previous_frame') and track.has_previous_frame:
    # 在这里添加处理速度计算的代码

                prev_ltrb = track.to_ltrb(previous=True)

            # 计算像素变化
                pixel_change_x = ltrb[0] - prev_ltrb[0]
                pixel_change_y = ltrb[1] - prev_ltrb[1]

            # 将像素变化映射到实际距离变化
                distance_change_x = pixel_change_x / pixels_per_meter
                distance_change_y = pixel_change_y / pixels_per_meter

            # 计算速度，返回上一个帧和当前帧之间的时间差，以秒为单位
                time_interval = track.get_time_interval()
                if time_interval > 0:
                    speed_x = distance_change_x / time_interval
                    speed_y = distance_change_y / time_interval
                    velocity= np.sqrt(speed_x**2+speed_y**2)
                else:
                    velocity = 0.0
### 从视频帧中截取目标图像
___在鼠标事件函数中加入右键点击事件___

    elif event == cv2.EVENT_RBUTTONDOWN:  # 检测右键点击事件
            nonlocal capture_screenshot
            capture_screenshot = True
### 将选中目标单独监控
___规定输出视频的参数___

    # 初始化视频写入对象
    fourcc_1 = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = 'single.mp4'
    output_fps = 8  # 设置输出视频的帧率
    output_size = (frame_width, frame_height)
    writer_1 = cv2.VideoWriter(output_filename, fourcc_1, output_fps, output_size)
___将目标被选中后的帧处理后组成的数据框写入视频___

    # 在循环内判断是否需要截图
        if capture_screenshot:
        # 在选定对象框内截取图像
            if selected_object_id != -1:
                ltrb = tracks[selected_object_id].to_ltrb()
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                captured_image_1 = frame[y1:y2, x1:x2]  # 截取图像
                frame_list.append(captured_image_1)  # 将图像添加到帧列表
        # 将每一帧的图像写入视频文件
        
       # 在循环之前，对捕获的帧进行调整大小
        output_frame_list = []
        for frame_img in frame_list:
            resized_frame = cv2.resize(frame_img, output_size)
            output_frame_list.append(resized_frame)
             # 将调整大小后的帧写入视频
        for resized_frame in output_frame_list:
            writer_1.write(resized_frame)
        output_frame_list.clear()
        frame_list.clear()
