import re
from argparse import ArgumentParser
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO

import gradio as gr

from pick_place_er1_trace import MotionPlanner

def _gc():
    import gc
    gc.collect()

def pil_to_base64(img):
    """将PIL图像转换为Base64编码的字符串"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def numpy_to_base64(img_array):
    """将numpy数组转换为Base64编码的HTML图像标签"""
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    img_base64 = pil_to_base64(img_pil)
    img_html = f'<img src="data:image/png;base64,{img_base64}" alt="Image" />'
    return img_html

def main():
    parser = ArgumentParser()

    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=2932, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    args = parser.parse_args()
    
    motion_planner = MotionPlanner()

    def reset_state(_chatbot, task_history, planner_state):
        task_history.clear()
        _chatbot.clear()
        planner_state.clear()
        _gc()
        return [], {}

    def init_message():
        welcome_msg = "这个Demo将展示使用ER1-Trace模型进行路径规划和执行。"
        welcome_msg2 = "请首先点击'拍摄图像'按钮，拍摄当前场景的图像。"
        return [(None, welcome_msg), (None, welcome_msg2)], [(None, welcome_msg), (None, welcome_msg2)]
    
    def capture_image_user_message(_chatbot, task_history):
        # 更新聊天界面
        _chatbot = _chatbot if _chatbot is not None else []
        task_history = task_history if task_history is not None else []
        
        # 保留命令记录，用于显示操作
        _chatbot.append(("📷 拍摄图像", None))
        task_history.append(("📷 拍摄图像", None))
        
        return _chatbot, task_history

    def capture_image(_chatbot, task_history, planner_state):
        """拍摄图像并初始化规划器"""
        try:
            # 拍摄图像
            if 'planner' not in planner_state:
                planner_state['planner'] = motion_planner
            
            planner = planner_state['planner']
            obs = planner.preexecute()
            img_rgb = np.asarray(obs["im_rgbd"].color)
            
            # 保存当前观测状态
            planner_state['obs'] = obs
            planner_state['img_rgb'] = img_rgb
            
            # 更新聊天界面
            _chatbot = _chatbot if _chatbot is not None else []
            task_history = task_history if task_history is not None else []
            
            # 将图像添加到聊天框中
            _chatbot.append((None, "✅ 图像拍摄完成！请在上方输入框中输入任务指令，然后点击'规划路径'按钮。"))
            # 将numpy数组转换为HTML图像标签
            img_html = numpy_to_base64(img_rgb)
            _chatbot.append((None, img_html))  # 显示拍摄的图像
            task_history.append(("图像拍摄", "完成"))
            
            return _chatbot, task_history, planner_state
            
        except Exception as e:
            error_msg = f"❌ 拍摄图像时出错: {str(e)}"
            _chatbot = _chatbot if _chatbot is not None else []
            _chatbot.append((None, error_msg))
            return _chatbot, task_history, planner_state

    def plan_path_user_message(_chatbot, task_history, task_instruction):
        # 更新聊天界面
        _chatbot = _chatbot if _chatbot is not None else []
        task_history = task_history if task_history is not None else []
        
        # 保留命令记录，用于显示操作
        _chatbot.append(("✍️ 规划路径", None))
        task_history.append(("✍️ 规划路径", None))
        
        _chatbot.append((task_instruction, None))
        task_history.append((task_instruction, None))
        
        return _chatbot, task_history

    def plan_path(_chatbot, task_history, planner_state, task_instruction):
        """规划路径"""
        try:
            if 'planner' not in planner_state or 'obs' not in planner_state:
                error_msg = "❌ 请先点击'拍摄图像'按钮！"
                _chatbot = _chatbot if _chatbot is not None else []
                _chatbot.append((None, error_msg))
                return _chatbot, task_history, planner_state
            
            if not task_instruction.strip():
                error_msg = "❌ 请输入任务指令！"
                _chatbot = _chatbot if _chatbot is not None else []
                _chatbot.append((None, error_msg))
                return _chatbot, task_history, planner_state
            
            planner = planner_state['planner']
            obs = planner_state['obs']
            
            # 规划路径
            trace, text_result, debug_image_rgb = planner.plan(task_instruction.strip(), obs)
            
            # 保存规划结果
            planner_state['trace'] = trace
            planner_state['debug_image'] = debug_image_rgb
            
            # 更新聊天界面
            _chatbot = _chatbot if _chatbot is not None else []
            task_history = task_history if task_history is not None else []
            
            # 更新聊天界面
            success_msg = f"✅ 路径规划完成！请点击'执行路径'按钮执行规划好的路径。任务: {task_instruction.strip()}\n{text_result.replace('<', '&lt;').replace('>', '&gt;')}"
            _chatbot.append((None, success_msg))
            # 将numpy数组转换为HTML图像标签
            debug_img_html = numpy_to_base64(debug_image_rgb)
            _chatbot.append((None, debug_img_html))  # 显示规划结果图像
            task_history.append((task_instruction.strip(), success_msg))
            
            return _chatbot, task_history, planner_state
            
        except Exception as e:
            error_msg = f"❌ 路径规划时出错: {str(e)}"
            _chatbot = _chatbot if _chatbot is not None else []
            _chatbot.append((None, error_msg))
            return _chatbot, task_history, planner_state

    def execute_path_user_message(_chatbot, task_history):
        # 更新聊天界面
        _chatbot = _chatbot if _chatbot is not None else []
        task_history = task_history if task_history is not None else []
        
        # 保留命令记录，用于显示操作
        _chatbot.append(("🚀 执行路径", None))
        task_history.append(("🚀 执行路径", None))
        
        return _chatbot, task_history

    def execute_path(_chatbot, task_history, planner_state):
        """执行路径"""
        try:
            if 'planner' not in planner_state or 'trace' not in planner_state:
                error_msg = "❌ 请先完成图像拍摄和路径规划！"
                _chatbot = _chatbot if _chatbot is not None else []
                _chatbot.append((None, error_msg))
                return _chatbot, task_history, planner_state
            
            planner = planner_state['planner']
            trace = planner_state['trace']
            
            # 执行路径
            planner.execute(trace)
            
            # 更新聊天界面
            _chatbot = _chatbot if _chatbot is not None else []
            task_history = task_history if task_history is not None else []
            
            # 更新聊天界面
            success_msg = "✅ 路径执行完成！机械臂已按照规划路径完成操作。"
            _chatbot.append((None, success_msg))
            task_history.append(("执行路径", success_msg))
            
            return _chatbot, task_history, planner_state
            
        except Exception as e:
            error_msg = f"❌ 路径执行时出错: {str(e)}"
            _chatbot = _chatbot if _chatbot is not None else []
            _chatbot.append((None, error_msg))
            return _chatbot, task_history, planner_state
    
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=6>🤔 Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation</center>""")

        chatbot = gr.Chatbot(label='操作日志与图像显示', elem_classes='control-height', height=600)
        query = gr.Textbox(lines=2, label='任务指令', placeholder="请输入任务指令，例如：Pick up the sponge and put it on the plate.")
        
        with gr.Row():
            img_btn = gr.Button('📷 拍摄图像', variant='primary')
            submit_btn = gr.Button('✍️ 规划路径', variant='secondary')
            execute_btn = gr.Button('🚀 执行路径', variant='stop')
            empty_bin = gr.Button('🧹 重新开始', variant='secondary')

        # 状态管理
        task_history = gr.State([])
        planner_state = gr.State({})

        # 按钮事件绑定
        img_btn.click(
            capture_image_user_message, 
            [chatbot, task_history], 
            [chatbot, task_history], 
            show_progress=True
        ).then(
            capture_image, 
            [chatbot, task_history, planner_state], 
            [chatbot, task_history, planner_state], 
            show_progress=True
        )
        
        submit_btn.click(
            plan_path_user_message, 
            [chatbot, task_history, query], 
            [chatbot, task_history], 
            show_progress=True
        ).then(
            plan_path, 
            [chatbot, task_history, planner_state, query], 
            [chatbot, task_history, planner_state], 
            show_progress=True
        )
        
        execute_btn.click(
            execute_path_user_message, 
            [chatbot, task_history], 
            [chatbot, task_history], 
            show_progress=True
        ).then(
            execute_path, 
            [chatbot, task_history, planner_state], 
            [chatbot, task_history, planner_state], 
            show_progress=True
        )
        
        empty_bin.click(
            reset_state, 
            [chatbot, task_history, planner_state], 
            [chatbot, planner_state], 
            show_progress=True
        ).then(
            init_message, 
            [], 
            [chatbot, task_history]
        )

        gr.Markdown("""⚠️⚠️⚠️ 机械臂使用可能存在夹手伤人风险，请在使用时确认环境安全，并由专业人员操作。""")
        
        demo.load(init_message, [], [chatbot, task_history])

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )

if __name__ == '__main__':
    main()