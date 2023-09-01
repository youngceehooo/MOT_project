# MOT_project
the repository to be submited as our summer camp project 
## 小组成员：
杨子赫-U202112240\
杨呈恺-U202112236\
詹智杰-U202112243\
吴炳旭-U202112233\
## 项目功能：
**_本项目用于多目标追踪场景，提供目标选择、计算目标速度、从视频帧中截取目标图像、将选中目标单独监控等功能_**\
### 目标选择功能：
___创建鼠标回测函数___
'''def on_mouse_click(event, x, y, flags, param):
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
'''cv2.setMouseCallback("Frame", on_mouse_click)  # 设置鼠标事件回调函数'''
___遍历跟踪器进行处理___
'''for idx, track in enumerate(tracks):
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
               '''