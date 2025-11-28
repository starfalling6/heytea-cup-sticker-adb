# 标准化导入方式
import os
import sys
import time
import math
import subprocess
from typing import Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ======================== 全局配置常量（抽离所有可配置项） ========================
# ADB相关配置
ADB_EXECUTABLE_PATH = r"B:\AveryDev\scrcpy-win64-v3.3.3\adb.exe"

# 绘图核心参数
DRAW_STEP = 5  # 跳行步长：每隔5行画一行
SWIPE_DURATION_MS = 150  # 滑动持续时间(毫秒)
MIN_LINE_LENGTH_PX = 8  # 最小线段长度(像素)
LINE_DRAW_INTERVAL_S = 0.2  # 每笔画完后的休息时间(秒)

# 画布布局配置
SCREEN_MARGIN_RATIO = 0.1  # 屏幕边距比例
MAX_CANVAS_HEIGHT_RATIO = 0.7  # 画布最大高度占屏幕比例
OFFSET_X_ADJUST = 60  # X轴偏移修正值
OFFSET_Y_ADJUST = -160  # Y轴偏移修正值

# 文件配置
TARGET_IMAGE_FORMATS = [
    ("image.png", "PNG格式图片"),
    ("image.jpg", "JPG格式图片")
]


# ======================== 工具函数 ========================
def execute_adb_command(cmd_arguments: list) -> str:
    """
    执行ADB命令并返回输出结果

    Args:
        cmd_arguments: ADB命令参数列表

    Returns:
        命令执行输出结果
    """
    try:
        full_command = [ADB_EXECUTABLE_PATH] + cmd_arguments
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print("❌ 错误：找不到 adb 可执行文件，请检查路径配置")
        sys.exit(1)


def get_device_screen_resolution() -> Tuple[Optional[int], Optional[int]]:
    """
    获取设备屏幕分辨率

    Returns:
        (宽度, 高度) 或 (None, None)
    """
    screen_info = execute_adb_command(["shell", "wm", "size"])
    if "Physical size:" in screen_info:
        try:
            resolution_part = screen_info.split(":")[-1].strip()
            width, height = map(int, resolution_part.split("x"))
            return width, height
        except (ValueError, IndexError):
            pass

    print("❌ 无法获取屏幕分辨率，请检查手机连接状态")
    return None, None


def draw_single_line(x1: float, y1: float, x2: float, y2: float) -> None:
    """
    通过ADB绘制单条线段

    Args:
        x1, y1: 起点坐标
        x2, y2: 终点坐标
    """
    # 计算线段长度
    line_length = math.hypot(x2 - x1, y2 - y1)

    # 确保线段长度不小于最小值
    if line_length < MIN_LINE_LENGTH_PX:
        x2 = x1 + MIN_LINE_LENGTH_PX

    # 执行绘制命令
    draw_cmd = [
        "shell", "input", "swipe",
        str(int(x1)), str(int(y1)),
        str(int(x2)), str(int(y2)),
        str(SWIPE_DURATION_MS)
    ]
    execute_adb_command(draw_cmd)
    time.sleep(LINE_DRAW_INTERVAL_S)


def find_target_image_path() -> Optional[str]:
    """
    查找当前目录下的目标图片文件

    Returns:
        图片路径或None
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))

    for filename, desc in TARGET_IMAGE_FORMATS:
        image_path = os.path.join(current_directory, filename)
        if os.path.exists(image_path):
            return image_path

    return None


# ======================== 核心绘图逻辑 ========================
def process_image_and_draw(image_path: str) -> None:
    """
    处理图片并通过ADB在设备上绘制

    Args:
        image_path: 图片文件路径
    """
    # 1. 检查设备连接状态
    print("🔍 正在检测Android设备连接...")
    device_list = execute_adb_command(["devices"])
    if "device" not in device_list.replace("List of devices attached", "").strip():
        print("❌ 未检测到Android设备，请开启USB调试并连接设备")
        return

    # 2. 获取屏幕分辨率
    screen_width, screen_height = get_device_screen_resolution()
    if not screen_width or not screen_height:
        return
    print(f"✅ 设备屏幕分辨率：{screen_width}x{screen_height}")

    # 3. 验证图片文件
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在：{image_path}")
        return

    # 4. 读取并预处理图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片文件：{image_path}")
        return

    # 5. 计算图片缩放参数
    screen_margin = int(screen_width * SCREEN_MARGIN_RATIO)
    initial_canvas_width = screen_width - 2 * screen_margin

    img_height, img_width = image.shape[:2]
    scale_factor = initial_canvas_width / img_width
    scaled_height = int(img_height * scale_factor)

    # 调整缩放比例以适应屏幕高度限制
    if scaled_height > screen_height * MAX_CANVAS_HEIGHT_RATIO:
        scale_factor = (screen_height * MAX_CANVAS_HEIGHT_RATIO) / img_height
        scaled_height = int(img_height * scale_factor)
        initial_canvas_width = int(img_width * scale_factor)
        screen_margin = (screen_width - initial_canvas_width) // 2

    scaled_width = initial_canvas_width
    print(f"🖼️ 图片缩放尺寸：{scaled_width}x{scaled_height}")

    # 6. 缩放图片并转换为二值图
    resized_image = cv2.resize(
        image,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA
    )
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # 7. 计算绘制偏移量
    draw_offset_x = screen_margin + OFFSET_X_ADJUST
    draw_offset_y = (screen_height - scaled_height) // 2 + OFFSET_Y_ADJUST

    # 8. 绘制前提示
    print("=" * 30)
    print("   🎨 进入ADB自动绘图模式")
    print(f"   步长：{DRAW_STEP}行 | 滑动时长：{SWIPE_DURATION_MS}ms")
    print(f"   间隔：{LINE_DRAW_INTERVAL_S}s | 最小线段：{MIN_LINE_LENGTH_PX}px")
    print("   ⚠️  请确保手机已打开画图软件，画布为空")
    print("=" * 30)
    print("⏳ 2秒后开始绘制...")
    time.sleep(2)

    # 9. 开始绘制
    total_lines_drawn = 0
    start_time = time.time()

    # 逐行绘制
    for row_idx in tqdm(
            range(0, scaled_height, DRAW_STEP),
            desc='📝 绘图进度',
            ncols=80
    ):
        row_data = binary_image[row_idx]
        col_idx = 0

        while col_idx < scaled_width:
            if row_data[col_idx] == 0:
                # 找到连续的绘制区域
                start_col = col_idx
                while col_idx < scaled_width and row_data[col_idx] == 0:
                    col_idx += 1
                end_col = col_idx - 1

                # 计算绘制坐标
                start_x = draw_offset_x + start_col
                start_y = draw_offset_y + row_idx
                end_x = draw_offset_x + end_col
                end_y = draw_offset_y + row_idx

                # 绘制线段
                draw_single_line(start_x, start_y, end_x, end_y)
                total_lines_drawn += 1
            else:
                col_idx += 1

    # 10. 绘制完成统计
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✅ 绘图完成！")
    print(f"📊 绘制统计：")
    print(f"   总线段数：{total_lines_drawn}")
    print(f"   总耗时：{elapsed_time:.2f}秒")
    print(f"   平均速度：{total_lines_drawn / elapsed_time:.2f}线段/秒")


# ======================== 主程序入口 ========================
def main() -> None:
    """主程序入口"""
    # 查找目标图片
    target_image_path = find_target_image_path()
    if not target_image_path:
        print("❌ 未找到目标图片文件（image.png 或 image.jpg）")
        sys.exit(1)

    print(f"📂 找到图片文件：{target_image_path}")

    # 执行绘图流程
    process_image_and_draw(target_image_path)


if __name__ == "__main__":
    main()