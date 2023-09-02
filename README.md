# MOT_project
the repository to be submited as our summer camp project 
## PPT链接
https://gamma.app/docs/MOTproject-kfgxvp34jjulx3j?mode=doc
## 各文件用途
trackers文件夹存储项目相关文件，\
based_yolonas_deepsort.ipynb为jupyter源文件格式的项目代码，\
16run.mp4,test.mp4,face.mp4均为输入文件可以通过更改项目代码中的输入文件名来更改要处理的视频，\
coco.names为yolo_nas_1模型检测后的标签文件用于标记类别，\
output.mp4为处理后的输出视频，\
single.mp4为锁定选中目标的输出视频，\
captured_image.jpg为选中目标的实时截图
## 环境搭建（注意版本兼容）
1.torch(2.0.1+cu117,2.0.1+cu118)\
2.cuda(11.7,11.8)\
3.super_gradients(3.2.0)\
4.visual studio编译c++
## 研究背景和项目目标
### 选题依据
1.丰富的开源资源支持，可以站在巨人的肩膀上实现想要的功能\
2.更加贴近现实场景，有助于功能拓展更加实用\
3.涉及的操作更加全面，有助于从中得到锻炼
### 业界现状介绍
1.深度学习应用\
2.端到端的追踪\
3.多传感器融合\
4.实时性要求\
5.长时间追踪\
6.广泛的应用领域\
7.开源工具和库\
8.评估和竞赛
### 本项目的目标
___可应用于多目标追踪场景，主要为监控视频的计算机视觉处理，提供目标选择、计算目标速度、从视频帧中截取目标图像、将选中目标单独监控等功能，
期待通过后续开发实现姿势预测算法触发报警功能,达成完备的监控安全系统，并将性能提升到计算机视觉处理实时性要求的5-30帧范围___
## 项目管理
### 小组成员：
杨子赫-U202112240\
杨呈恺-U202112236\
詹智杰-U202112243\
吴炳旭-U202112233
### 任务分工：
杨子赫：环境搭建，git操作，代码编辑\
杨呈恺：代码编辑，报告编写\
詹智杰：代码编辑，ppt制作\
吴炳旭：代码编辑，资料检索
## 项目总体设计
### 团队架构
我们采用的团队协作方式为支流合并为主流，利用git系统在仓库中设立分支，每个人在自己的分支上修改和维护代码，开展工作，这样分支不会影响到主干的正常运行，最后利用git merge合并分支汇总到master分支，同时我们在Slack应用上创建了工作空间，运用人工智能Claude辅助了团队协作和交流
### 项目工具包
1.anaconda进行环境管理和包管理 https://www.anaconda.com/ \
2.jupyter notebook为anaconda自带的IDLE平台，更加轻量化\
3.PyTorch是一个深度学习框架，它是一个开源的机器学习库，广泛用于深度神经网络的研究和应用开发 https://pytorch.org \
4.super_gradients框架里面包含了deepsort、yolo_nas等先进的算法模型 https://www.supergradients.com/ \
5.opencv是一个广泛用于计算机视觉和图像处理的开源库 
    
    pip install opencv
    conda install opencv
### 项目分解
1.环境配置\
2.基础功能（目标检测+跟踪器）代码\
3.各分支分配功能模块\
4.合并主函数\
5.编写报告
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

## 项目功能代码（项目总体设计）：
**_本项目用于多目标追踪场景，提供目标选择、计算目标速度、从视频帧中截取目标图像、将选中目标单独监控等功能_**\
___以下代码均省略初始化参数___
### 目标选择功能：
___创建鼠标回测函数___
   
    ___def on_mouse_click(event, x, y, flags, param):        
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
## 项目测试
### 功能实现
___预期功能基本完成，测量速度模块需要根据像素对应现实距离进行映射参数pixels_per_meter的调整___
___获得output.mp4,single.mp4,captured_image三个文件___
### 性能优化
对于一些计算机视觉和图像处理任务，如实时目标检测、跟踪、姿态估计等，通常需要较高的FPS，以确保实时性。一般要求在每秒5帧到每秒30帧之间，具体取决于任务的复杂性。\
当只进行单目标跟踪时，即只创建一个跟踪器，cpu处理速度可达30帧以上，
当创建多个跟踪器并进行目标检测，cpu处理速度只有不到1FPS，而GPU处理速度在4FPS左右
## 总结与反思
经过本次项目，我们对于计算机视觉处理有了基本了解，提升了编程能力，对于git系统的操作更加熟练，运用网络检索资源的能力提高，最为重要的是深刻体会了人工智能的力量，面向chatgpt编程不失为一种好手段:laughing:


