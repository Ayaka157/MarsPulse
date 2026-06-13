"""
MarsPulse — 行空板嵌入式演示主程序
==================================
模块说明: 在UniHiker行空板上运行的火星基地资源调配实物演示。
         加载导出的NumPy策略权重，实现三智能体前向推理。
         驱动WS2812B LED灯带(蓝=氧气流动/红=电力供应)和SSD1306 OLED屏幕
         实时显示S(t)指数与Agent动作。支持按钮触发"沙尘暴模式"。
硬件: UniHiker, WS2812B LED灯带, SSD1306 OLED 128×64, 按钮模块
作者: 陈凯熙
开发时间: 2026年5月
"""

# -*- coding: utf-8 -*-
"""MarsPulse 行空板 M10 横屏演示程序 (特定资源颜色绑定版).

硬件交互行为说明：
  - 外接模拟光敏传感器 (接 P22 引脚)：用手遮挡时模拟太阳辐射骤减 -> 触发火星沙尘暴模式。
  - 外接数字按键开关 (接 P23 引脚)：按下时输入低电平 0 -> 触发生命保障系统故障（氧气瞬间暴跌 100 kg）。
  - 外接 WS2812B 智能灯带 (接 P21 引脚)：分段流水灯显示，颜色根据特定区域资源状态动态渐变演变。
  - AI 决策机制：直接从导出的 NumPy 权重文件读取并运行，无需在行空板上部署笨重的 PyTorch 深度学习框架。
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

# 尝试导入 PIL 图像处理库（行空板屏幕渲染核心依赖），若在普通 PC 终端测试无 GUI 环境则优雅降级
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


# --- 路径与全局硬件引脚配置 ---
BASE_DIR = os.path.dirname(__file__)
POLICY_PATH = os.path.join(BASE_DIR, "mappo_policy_unihiker.npz")  # 本地 AI 策略网络权重路径
FRAME_PATH = "/tmp/marspulse_m10_landscape.png"                    # 缓存屏幕帧的临时路径

LED_PIN_NAME = os.environ.get("MARSPULSE_LED_PIN", "P21")           # 灯带数据引脚
EXTERNAL_LIGHT_PIN_NAME = os.environ.get("MARSPULSE_LIGHT_PIN", "P22") # 光敏传感器引脚
EXTERNAL_BUTTON_PIN_NAME = os.environ.get("MARSPULSE_BUTTON_PIN", "P23") # 故障模拟按键引脚

# --- WS2812B 灯带区域划分 ---
LED_COUNT = 14
LIVING_ZONE = (0, 2)  # 第 1 区域 (居住舱)：占用 Index 0-2 (共 3 颗 LED)
POWER_ZONE = (3, 7)   # 第 2 区域 (储能中心)：占用 Index 3-7 (共 5 颗 LED)
LSS_ZONE = (8, 13)    # 第 3 区域 (生命保障区)：占用 Index 8-13 (共 6 颗 LED)

# --- 屏幕分辨率与显示刷新控制 ---
LANDSCAPE_W = 320    # 横屏画布逻辑宽度
LANDSCAPE_H = 240    # 横屏画布逻辑高度
PHYSICAL_W = 240     # 行空板物理屏幕实际宽度
PHYSICAL_H = 320     # 行空板物理屏幕实际高度

SCREEN_ROTATION_DEG = 90    # 旋转 90 度以适配行空板的竖长屏物理结构
SCREEN_REFRESH_TICKS = 2    # 降低屏幕刷新频率（每 2 个主循环 Tick 刷新一次），为 AI 留出算力

LIGHT_STORM_THRESHOLD = int(os.environ.get("MARSPULSE_LIGHT_THRESHOLD", "80")) # 沙尘暴触发的光敏阈值
LED_SELF_TEST_SECONDS = 1.0 # 开机硬件自检时灯带亮起的时间
LOOP_DELAY = 0.05           # 主循环基础延时（50 毫秒/次）
SIM_HOURS_PER_LOOP = 0.5    # 每次循环在物理模拟器中推进的火星时间（0.5 小时）


class MarsPulseState:
    """火星基地核心状态机：记录所有物理环境、资源余量、硬件采样以及 AI 决策数据"""
    def __init__(self):
        self.time_sol = 0.0          # 火星日计时 (Sol)
        self.energy_kwh = 2400.0     # 蓄电池剩余电量 (满载 3000 kWh)
        self.oxygen_kg = 450.0       # 基础氧气存量 (满载 500 kg)
        self.temp_c = 20.0           # 基地内部核心温度 (摄氏度)
        self.tau = 0.15              # 大气光学厚度 (Tau，数值越大说明沙尘暴越严重，阳光越少)
        self.pin_kw = 0.0            # 当前太阳能板/RTG 实时输入功率 (kW)
        self.survival_index = 1.0    # 综合生存指数 S(t)，范围 0.0 ~ 1.0
        self.storm_active = False    # 沙尘暴状态旗标
        self.oxygen_leak = False     # 氧气泄漏故障状态旗标
        self.fault_triggered = False  # 故障单次扣血锁扣（确保按键按下时只扣一次大分，不持续暴扣）
        self.failures = [0.0, 0.0, 0.0, 0.0] # 传给 AI 的故障状态向量
        self.actions = np.array([12.0, 5.0, 10.0], dtype=np.float32) # 初始化动作 [生保供电, 加热供电, 科学实验供电]
        self.mode_text = "NORMAL"    # 状态文本标签
        self.ai_text = "AI INIT"      # AI 决策意图文本描述
        self.light_value = 0         # 实时读取的光敏电阻模拟值
        self.button_value = -1       # 按键当前的实时电平值
        self.button_idle = -1        # 空闲电平参考
        self.fault_drop_accum = 0.0  # 故障触发后的秒级持续失血积分器


class NumpyMAPPOPolicy:
    """基于纯 NumPy 数组前向传播的多智能体 AI 决策网络 (MAPPO)"""
    def __init__(self, path: str):
        self.loaded = False
        self.arrays = None
        if os.path.exists(path):
            # 加载并解析打包好的权重文件（包含各智能体的权重与偏置矩阵）
            self.arrays = np.load(path, allow_pickle=False)
            self.loaded = True

    def _forward_actor(self, agent_id: int, obs: np.ndarray) -> float:
        """底层多层感知机 (MLP) 的前向矩阵乘法：输入环境状态，输出最优供电动作"""
        a = self.arrays
        prefix = f"a{agent_id}_"
        x = np.asarray(obs, dtype=np.float32)
        
        # 两层隐藏层，使用 tanh 作为激活函数
        x = np.tanh(a[prefix + "net_0_weight"].dot(x) + a[prefix + "net_0_bias"])
        x = np.tanh(a[prefix + "net_2_weight"].dot(x) + a[prefix + "net_2_bias"])
        
        # 输出层（线性输出后进行归一化及缩放）
        raw = a[prefix + "net_4_weight"].dot(x) + a[prefix + "net_4_bias"]
        mean = np.tanh(raw)
        
        # 将网络输出的反归一化到实际物理动作空间 [Low, High]
        action = mean * a[prefix + "action_scale"] + a[prefix + "action_bias"]
        lo = float(a[prefix + "action_low"])
        hi = float(a[prefix + "action_high"])
        return float(np.clip(action[0], lo, hi))

    def predict(self, state: MarsPulseState) -> np.ndarray:
        """主预测接口：抽取状态机的指标进行归一化，分发给 3 个智能体并行决策"""
        if not self.loaded:
            return self._fallback_policy(state)  # 容错：如果权重文件不存在，切换至人工专家规则

        # 数据归一化特征工程，确保数值落在神经网络喜欢的区间内
        energy_frac = clamp01(state.energy_kwh / 3000.0)
        oxygen_frac = clamp01(state.oxygen_kg / 500.0)
        temp_norm = (state.temp_c - 5.0) / 30.0  # 温度标准化
        storm = 1.0 if state.storm_active else 0.0
        tau_norm = min(state.tau / 2.5, 2.0)
        failures = list(state.failures)

        # 构建 3 个子智能体各自的局部观测空间 (Observation Space)
        obs0 = np.array([energy_frac, oxygen_frac, storm, tau_norm] + failures, dtype=np.float32) # 智能体 0 关注生命保障
        obs1 = np.array([temp_norm, energy_frac, storm] + failures, dtype=np.float32)            # 智能体 1 关注热力控制
        obs2 = np.array([energy_frac, storm, oxygen_frac] + failures, dtype=np.float32)          # 智能体 2 关注科学实验

        # 并行执行网络推理，打包成 NumPy 动作数组返回
        return np.array(
            [
                self._forward_actor(0, obs0), # 输出生命保障系统供电功率 (kW)
                self._forward_actor(1, obs1), # 输出中央加温系统供电功率 (kW)
                self._forward_actor(2, obs2), # 输出科学实验项目供电功率 (kW)
            ],
            dtype=np.float32,
        )

    def _fallback_policy(self, state: MarsPulseState) -> np.ndarray:
        """专家硬编码兜底策略：在没有 AI 权重文件时保证系统不会死机"""
        energy_frac = state.energy_kwh / 3000.0
        oxygen_frac = state.oxygen_kg / 500.0
        
        if state.oxygen_leak or oxygen_frac < 0.35:
            return np.array([22.0, 5.0, 0.0], dtype=np.float32)  # 极度缺氧：强开生保，熔断科学实验
        if state.storm_active or energy_frac < 0.30:
            return np.array([12.0, 3.0, 0.0], dtype=np.float32)  # 沙尘暴缺电：压低能耗维持基本运转
        return np.array([14.0, 6.0, 14.0], dtype=np.float32)     # 资源充沛：全面均衡发展


class UniHikerIO:
    """硬件驱动与 GUI 渲染层：专门负责行空板的物理外设读取、显示屏绘制及灯带控制"""
    def __init__(self):
        self.gui = None
        self.image_widget = None
        self.np_strip = None
        self.hardware_ok = False
        self.screen_tick = -1
        self.font_big = None
        self.font_mid = None
        self.font_small = None
        self.last_button_value = -1
        
        self.power_offset = 0
        self.water_offset = 0
        self.speed_accumulator = 0.0  # 流水灯移动速度累加器

        try:
            # 动态导入行空板专属的底层 Python 硬件库 unihiker 和 pinpong
            from unihiker import GUI
            from pinpong.board import Board, Pin, ADC
            import pinpong.board as pb

            # 初始化行空板板载芯片
            Board("UNIHIKER").begin()
            self.Pin = Pin
            self.gui = GUI()

            # 初始化外部硬件接口
            led_pin = getattr(Pin, LED_PIN_NAME)
            self.np_strip = pb.NeoPixel(Pin(led_pin), LED_COUNT) # 驱动外部 WS2812B 灯带
            self.np_strip.brightness(95)                         # 设置较高亮度
            
            light_pin = getattr(Pin, EXTERNAL_LIGHT_PIN_NAME)
            button_pin = getattr(Pin, EXTERNAL_BUTTON_PIN_NAME)
            self.light_adc = ADC(Pin(light_pin))                 # 注册光敏模拟输入引脚
            
            # 关键：将 P23 设为上拉输入模式 (Pull-Up)。空闲时为 1，按下接地时为 0
            self.fault_button = Pin(button_pin, Pin.IN, Pin.PULL_UP)
            
            time.sleep(0.05)
            self.last_button_value = 1
            self.hardware_ok = True

            self._load_fonts()   # 加载系统字体
            self._init_screen()  # 初始化横屏布局
            self._led_self_test() # 执行开机炫彩灯光自检
        except Exception as exc:
            print("Hardware init failed, console mode:", exc)

    def _load_fonts(self) -> None:
        """寻找行空板 Debian 系统内置的 TrueType 字体以渲染精美的 UI"""
        if ImageFont is None:
            return
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        normal = next((p for p in font_paths if os.path.exists(p)), None)
        bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(bold):
            bold = normal
        try:
            self.font_big = ImageFont.truetype(bold, 30)
            self.font_mid = ImageFont.truetype(bold, 16)
            self.font_small = ImageFont.truetype(normal, 11)
        except Exception:
            self.font_big = ImageFont.load_default()
            self.font_mid = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def _init_screen(self) -> None:
        """清屏并建立全局唯一的图片承载组件，后续通过不停覆盖更新该 png 路径实现极速刷新"""
        if self.gui is None:
            return
        self.gui.clear()
        self.gui.fill_rect(x=0, y=0, w=PHYSICAL_W, h=PHYSICAL_H, color="#07111F")
        self._save_frame(MarsPulseState())
        self.image_widget = self.gui.draw_image(
            x=PHYSICAL_W / 2, y=PHYSICAL_H / 2, image=FRAME_PATH, origin="center"
        )

    def read_light(self) -> int:
        """读取 P22 引脚上模拟光敏电阻的电压离散值"""
        if not self.hardware_ok:
            return 3000
        try:
            return int(self.light_adc.read()) # 返回 0-4095 区间值，数值越小环境越暗
        except Exception:
            return 3000

    def read_fault_button(self) -> bool:
        """优化防抖数字滤波算法：严格杜绝引脚悬空或长线接触引发的电平瞬时毛刺噪声"""
        if not self.hardware_ok:
            return False
        try:
            v1 = int(self.fault_button.value())
            self.last_button_value = v1
            
            # 若触发了低电平信号 (注：原代码写 v1 == 1 触发，意味着你可能在主循环或者传感器接法上有特定定义。这里为你维护原逻辑)
            if v1 == 1:
                print(f"[DEBUG 侦測] {EXTERNAL_BUTTON_PIN_NAME} 變為低電平 0！正在執行防抖...")
                time.sleep(0.003) # 硬件级微秒延时采样
                v2 = int(self.fault_button.value())
                time.sleep(0.003)
                v3 = int(self.fault_button.value())
                
                # 只有连续三次采样全部维持高/低电平，才判定为物理有效按压
                if v2 == 1 and v3 == 1:
                    print(f"=================================================")
                    print(f"[🔔 HARDWARE ALERT] 外部實體按鈕 {EXTERNAL_BUTTON_PIN_NAME} 確認觸發成功！")
                    print(f"=================================================")
                    return True
                else:
                    print(f"[DEBUG 濾波] 判定為瞬間雜訊：v1=0, v2={v2}, v3={v3}")
            return False
        except Exception as e:
            print(f"[DEBUG ERROR] 讀取按鈕引腳異常: {e}")
            return False

    def update_screen(self, state: MarsPulseState, tick: int) -> None:
        """双缓冲画布刷新：将内存中绘制好的全量火星监控仪表盘推送到行空板屏幕上"""
        if tick == self.screen_tick:
            return
        if tick % SCREEN_REFRESH_TICKS != 0 and self.screen_tick >= 0:
            return # 跳帧控制
        self.screen_tick = tick

        if not self.hardware_ok or self.gui is None:
            # 纯控制台调试降级打印
            print(
                "mode={} S={:.0%} E={:.0f} O2={:.0f} T={:.1f} Pin={:.1f}".format(
                    state.mode_text, state.survival_index, state.energy_kwh, state.oxygen_kg, state.temp_c, state.pin_kw
                )
            )
            return

        self._save_frame(state)
        try:
            self.image_widget.config(image=FRAME_PATH) # 直接用临时帧刷新核心画布组件
        except Exception:
            self.gui.clear()
            self.image_widget = self.gui.draw_image(
                x=PHYSICAL_W / 2, y=PHYSICAL_H / 2, image=FRAME_PATH, origin="center"
            )

    def _save_frame(self, state: MarsPulseState) -> None:
        """基于 PIL 绘制横屏原始数据，随后将其强制逆时针或顺时针旋转 90 度以符合物理屏格式"""
        if Image is None:
            return
        img = Image.new("RGB", (LANDSCAPE_W, LANDSCAPE_H), "#07111F")
        draw = ImageDraw.Draw(img)
        self._draw_dashboard(draw, state) # 绘制全套科幻风仪表盘组件
        rotated = img.rotate(SCREEN_ROTATION_DEG, expand=True) # 旋转 90 度
        rotated.save(FRAME_PATH) # 保存为文件，供给 GUI 组件读取

    def _draw_dashboard(self, draw, state: MarsPulseState) -> None:
        """UI 美学排版引擎：利用方块、进度条及胶囊标签拼接出深色科技风仪表盘"""
        s = clamp01(state.survival_index)
        # 根据基地综合生存度智能切换警示色 (健康=绿色，警告=黄色，崩溃危险=红色)
        status_color = "#7EE787" if s >= 0.8 else ("#F4D35E" if s >= 0.5 else "#FF5555")
        mode = "FAULT" if state.oxygen_leak else ("STORM" if state.storm_active else "NORMAL")

        # 1. 顶栏绘制
        draw.rectangle((0, 0, LANDSCAPE_W, LANDSCAPE_H), fill="#07111F")
        draw.rectangle((0, 0, LANDSCAPE_W, 35), fill="#0E1B2E")
        draw.text((10, 7), "MarsPulse", font=self.font_mid, fill="#F5F7FA")
        draw.text((130, 10), "AI Resource Control", font=self.font_small, fill="#A7B4C8")
        self._pill_tag(draw, 248, 7, 62, 20, mode, status_color) # 绘制右上角当前运行模式标签

        # 2. 左侧大转盘：综合生存指数显示区域
        draw.text((16, 55), "SURVIVAL", font=self.font_small, fill="#A7B4C8")
        draw.text((16, 72), "{:>3.0f}%".format(s * 100), font=self.font_big, fill=status_color)
        self._bar(draw, 16, 112, 126, 14, s, status_color)    # 绘制主指标进度条
        draw.text((16, 130), "light {:4d}   thr {:3d}".format(state.light_value, LIGHT_STORM_THRESHOLD),
                  font=self.font_small, fill="#A7B4C8")

        # 3. 右侧四大支柱资源子面板 (能源、氧气、热量、太阳能输入)
        self._resource_panel(draw, 156, 48, "ENERGY", state.energy_kwh / 3000.0,
                             "{:.0f} kWh".format(state.energy_kwh), "#F4D35E")
        self._resource_panel(draw, 156, 90, "OXYGEN", state.oxygen_kg / 500.0,
                             "{:.0f} kg".format(state.oxygen_kg), "#5DADEC")
        self._resource_panel(draw, 156, 132, "TEMP", thermal_score(state.temp_c),
                             "{:.1f} C".format(state.temp_c), "#FF8C42")
        self._resource_panel(draw, 156, 174, "SOLAR", min(state.pin_kw / 130.0, 1.0),
                             "{:.1f} kW".format(state.pin_kw), "#7EE787")

        # 4. 底栏：实时输出 AI 决策分配的瓦数分配动作参数
        draw.rectangle((12, 178, 150, 218), outline="#1E293B", fill="#0B1626")
        draw.text((20, 184), "ACTIONS kW", font=self.font_small, fill="#A7B4C8")
        draw.text(
            (20, 201),
            "LS {:04.1f} H {:04.1f} SC {:04.1f}".format(
                state.actions[0], state.actions[1], state.actions[2]
            ),
            font=self.font_small,
            fill="#F5F7FA",
        )
        # 最底部滚动显示当前 AI 脑海中的思考决策文本
        draw.text((12, 220), state.ai_text[:45], font=self.font_small, fill=status_color)

    def _pill_tag(self, draw, x, y, w, h, text, color):
        """辅助函数：画一个带有彩色边框的胶囊圆角矩形状态标签"""
        draw.rectangle((x, y, x + w, y + h), fill="#101F33", outline=color)
        self._text_center(draw, (x, y, x + w, y + h), text, self.font_small, color)

    def _resource_panel(self, draw, x, y, label, value, value_text, color):
        """辅助函数：画一个标准的右侧资源看板单元，内嵌微型进度条"""
        draw.rectangle((x, y, x + 148, y + 34), fill="#0B1626", outline="#1E293B")
        draw.text((x + 8, y + 5), label, font=self.font_small, fill="#A7B4C8")
        draw.text((x + 82, y + 5), value_text, font=self.font_small, fill="#F5F7FA")
        self._bar(draw, x + 8, y + 22, 132, 6, value, color)

    def _bar(self, draw, x, y, w, h, value, color):
        """辅助函数：标准的比例进度条绘制器"""
        draw.rectangle((x, y, x + w, y + h), fill="#1E293B")
        draw.rectangle((x, y, x + int(w * clamp01(value)), y + h), fill=color)

    def _text_center(self, draw, box, text, font, fill):
        """辅助函数：计算字体边界大小，实现容器内文本绝对水平垂直居中"""
        x1, y1, x2, y2 = box
        try:
            tw, th = draw.textsize(text, font=font)
        except Exception:
            tw, th = (len(text) * 6, 10) # 兼容老旧版本的降级估算
        draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=font, fill=fill)

    def update_leds(self, state: MarsPulseState, tick: int) -> None:
        """核心重构部分：根据不同的舱室职能，将硬件 WS2812B 物理灯带绑定至特定的资源量上"""
        if not self.hardware_ok or self.np_strip is None:
            return

        # ⚡ 极恶惩罚：若蓄电池电量完全归零归干，直接把基地拉闸断电，灯带全灭！
        if state.energy_kwh <= 0:
            self.np_strip.clear()
            return

        s_val = clamp01(state.survival_index)
        is_crisis = state.storm_active or (state.energy_kwh / 3000.0 < 0.35)

        # ⚡ 动感流水灯速率机制：流水灯奔跑速度完全跟随综合生存度 S(t) 的 1.5 次方动态演变。环境越差跑得越无力
        speed_step = 0.15 + 0.85 * (s_val ** 1.5)
        self.speed_accumulator += speed_step
        
        if self.speed_accumulator >= 1.0:
            self.power_offset = (self.power_offset + 1) % self._zone_len(POWER_ZONE)
            self.water_offset = (self.water_offset + 1) % self._zone_len(LSS_ZONE)
            self.speed_accumulator = 0.0

        # --- 1. 第一间：居住舱 (LIVING_ZONE) 状态灯光渲染 ---
        if state.oxygen_leak:
            self._set_range(LIVING_ZONE, (0, 0, 0))       # 严重空气泄露时，居住舱为了安全执行彻底断电全黑
        elif state.storm_active:
            self._set_range(LIVING_ZONE, (125, 20, 0))    # 外部沙尘暴席卷时，亮起低沉的避难橙红色光
        else:
            self._set_range(LIVING_ZONE, (0, 40, 0))      # 正常状态下，常亮温馨舒适的微光绿

        # --- 2. 独立资源色彩渐变绑定算法核心 ---
        energy_frac = clamp01(state.energy_kwh / 3000.0)
        oxygen_frac = clamp01(state.oxygen_kg / 500.0)

        # ① 第二间 (储能中心)：颜色仅跟【剩余电量】紧密挂钩！
        # 满电时呈现标准亮黄色 (170, 120, 0)；随着电量衰减，绿光通道被无情抽干，10%电量以下彻底退化成血红色 (255, 0, 0)
        p_factor = 1.0 - clamp01((energy_frac - 0.1) / 0.8)  # 在电量 90% 到 10% 敏感动态滑移区间
        p_r = int(170 + (255 - 170) * p_factor)
        p_g = int(120 - 120 * p_factor)
        power_color = (p_r, p_g, 0) if not is_crisis else (int(p_r * 0.3), int(p_g * 0.3), 0) # 基地陷入危机时整体灯带亮度压低 70% 节能

        # ② 第三间 (生命保障区)：颜色仅跟【当前氧气余量】紧密挂钩！
        # 氧气满时呈现科幻青蓝色 (0, 160, 210)；缺氧时红色暴增、蓝绿耗尽，无氧时彻底沦为纯血红 (255, 0, 0)
        o_factor = 1.0 - clamp01((oxygen_frac - 0.1) / 0.8)  # 在氧气 90% 到 10% 敏感动态滑移区间
        l_r = int(0 + 255 * o_factor)       
        l_g = int(160 - 160 * o_factor)     
        l_b = int(210 - 210 * o_factor)     
        lss_color = (l_r, l_g, l_b) if not is_crisis else (int(l_r * 0.3), int(l_g * 0.3), int(l_b * 0.3))

        # --- 3. 实体硬件驱动物理脉冲输出 ---
        # 渲染第二间 (储能中心)：执行基于电量色彩的单点循环流水灯
        self._draw_single_dot_flow(POWER_ZONE, self.power_offset, power_color, (0, 0, 0))

        # 渲染第三间 (生命保障区)：
        if state.oxygen_leak:
            # 外部物理按键按下触发故障大漏气时，强行切断常规流水，提升至最高优先级的红蓝警车高频爆闪！
            flash_color = (255, 0, 0) if tick % 2 == 0 else (0, 0, 255)
            self._set_range(LSS_ZONE, flash_color)
        else:
            # 正常或沙尘暴时，执行伴随『氧气纯粹变红』的单点呼吸青/红流水
            self._draw_single_dot_flow(LSS_ZONE, self.water_offset, lss_color, (0, 0, 0))

    def _draw_single_dot_flow(self, zone, offset: int, active_color, background_color) -> None:
        """辅助函数：在指定范围的 LED 灯段中，渲染一颗正在奔跑的单点高亮像素"""
        start, end = zone
        for i in range(start, end + 1):
            relative_pos = i - start
            if relative_pos == offset:
                self._set_led(i, active_color)      # 流水灯高亮头部
            else:
                self._set_led(i, background_color) # 背景暗色

    def _set_led(self, index: int, color) -> None:
        """安全设定单颗 WS2812B 像素点，将 RGB 强制拦截在 0-255 安全范围内"""
        r, g, b = [int(max(0, min(255, c))) for c in color]
        self.np_strip[index] = (r, g, b)

    def _set_range(self, zone, color) -> None:
        """批量安全填充某灯带区间的颜色"""
        r, g, b = [int(max(0, min(255, c))) for c in color]
        start_index, end_index = zone
        self.np_strip.range_color(start_index, end_index + 1, self._color_hex(r, g, b))

    def _zone_len(self, zone) -> int:
        return zone[1] - zone[0] + 1

    def _color_hex(self, r: int, g: int, b: int) -> int:
        """将 RGB 三色整合压缩成单路 24位 16进制整型，用作底层驱动输入"""
        return (r << 16) | (g << 8) | b

    def _led_self_test(self) -> None:
        """开机硬件测试：让三间舱室分别亮起绿、黄、青基础底色，一秒后熄灭，确认灯带物理完好"""
        if self.np_strip is None:
            return
        try:
            self.np_strip.clear()
            thirds = [LIVING_ZONE, POWER_ZONE, LSS_ZONE]
            colors = [(0, 80, 20), (160, 110, 0), (0, 130, 190)]
            for seg, color in zip(thirds, colors):
                self._set_range(seg, color)
            time.sleep(LED_SELF_TEST_SECONDS)
            self.np_strip.clear()
        except Exception as exc:
            print("LED self-test failed:", exc)


def clamp01(value: float) -> float:
    """极其常用且关键的数值截断工具：保证任何数学运算不会越界超出 [0.0, 1.0] 概率区间"""
    return max(0.0, min(1.0, float(value)))


def thermal_score(temp_c: float) -> float:
    """热力舒适度打分器：以 20.0 度为人类完美极佳舒适区，每偏离 15 度舒适度线性扣减"""
    if temp_c >= 20.0:
        return clamp01(1.0 - (temp_c - 20.0) / 15.0)
    return clamp01(1.0 - (20.0 - temp_c) / 15.0)


def update_physics(state: MarsPulseState, actions: np.ndarray) -> None:
    """火星基地高拟真一阶微积分物理动态仿真引擎"""
    # 动作裁剪锁定，防止神经网络发疯输出违背物理定律的超大功率
    state.actions = np.clip(actions.astype(np.float32), [0.0, 0.0, 0.0], [30.0, 20.0, 25.0])

    pmax = 150.0      # 太阳能板正午理论峰值功率 (kW)
    rtg = 10.0        # 核电池 (RTG) 全天候死守保底输入功率 (kW)
    solar_eff = 0.85  # 太阳能转换率
    # 光学厚度指数衰减公式：沙尘暴越强 (tau 变大)，太阳能发电输入呈指数级断崖暴跌
    state.pin_kw = pmax * solar_eff * math.exp(-state.tau) + rtg

    life_support, heater, experiments = [float(x) for x in state.actions]
    total_draw = life_support + heater + experiments + 7.0 # 汇总当前总耗电（加上 7 kW 基地基础静态维持能耗）
    dt_h = SIM_HOURS_PER_LOOP # 推进的时间片步长

    # 1. 蓄电池电量迭代运算：剩余电量 = 净输入输出功率 * 时间
    state.energy_kwh += (state.pin_kw - total_draw) * dt_h
    state.energy_kwh = max(0.0, min(3000.0, state.energy_kwh))

    # 2. 氧气动力学生产/消耗迭代运算
    base_consumption = 1.25 if not state.storm_active else 12.5 # 沙尘暴恐慌情绪下，宇航员新陈代谢急剧飙升 10 倍！
    o2_consumption = base_consumption * 6.0 * (dt_h / 24.65)
    
    o2_efficiency = 0.03 if state.storm_active else 0.15 # 沙尘暴期间沙尘堵塞过滤格，制氧机效率打 2 折
    o2_prod = life_support * 0.6 * dt_h * o2_efficiency
    
    state.oxygen_kg += o2_prod - o2_consumption
    
    # --- 故障响应物理分支 ---
    if state.oxygen_leak:
        if not state.fault_triggered:
            state.oxygen_kg -= 100.0       # 物理触发瞬间，第一下直接爆裂性泄露扣除 100 kg 大分！
            state.fault_triggered = True  
            print(f"[💥 EFFECT] 物理引擎變量大扣除：state.oxygen_kg 瞬間減少 100 kg！")
        
        # 随后只要按键一直没放开，进入时间步积分：以每秒乘以 0.88 的速度呈指数几何级疯狂窒息性衰减
        state.fault_drop_accum += LOOP_DELAY
        while state.fault_drop_accum >= 1.0:
            state.oxygen_kg *= 0.88
            state.fault_drop_accum -= 1.0
    else:
        state.fault_triggered = False     
        state.fault_drop_accum = 0.0

    state.oxygen_kg = max(0.0, min(500.0, state.oxygen_kg))

    # 3. 基地热传导温度变迁迭代运算
    # 内部最终平衡目标温度取决于加热器供电和基地设备总散热量
    thermal_target = 20.0 + (heater - 5.0) * 0.05 + total_draw * 0.002
    if state.storm_active:
        thermal_target -= 1.5 # 沙尘暴带走热量
    if state.oxygen_leak:
        thermal_target -= 0.8 # 空气漏了导致热对流失衡
    # 模拟牛顿冷却定律：室内温度逐渐向目标温度逼近过度
    state.temp_c += (thermal_target - state.temp_c) * 0.08

    # 4. 汇总得出全新的最终 S(t) 生存指数评分（各项权重：电量 40%，氧气 35%，温度 25%）
    e = clamp01(state.energy_kwh / 3000.0)
    o = clamp01(state.oxygen_kg / 500.0)
    t = thermal_score(state.temp_c)
    state.survival_index = clamp01(0.40 * e + 0.35 * o + 0.25 * t)
    state.time_sol += dt_h / 24.65 # 更新火星日时间轴


def describe_ai(state: MarsPulseState) -> str:
    """意图解释器：将多智能体宏观当前的联合供电倾向，翻译成直观的监控中文字符串"""
    e = state.energy_kwh / 3000.0
    o = state.oxygen_kg / 500.0
    if state.oxygen_leak:
        return "AI: life-support fault, boost O2 and cut science" # 发生大漏气：AI 全力拉满制氧并熔断一切科学耗电
    if state.storm_active or e < 0.35:
        return "AI: dust storm, non-core zones powered down"     # 遭遇沙尘暴或低电量：关闭非核心区域执行避难节能
    if o < 0.45:
        return "AI: low oxygen, life support has priority"        # 氧气低：生命保障系统获取绝对供电优先权
    return "AI: balanced science and survival mode"               # 一切安好：生存与科研全面大踏步并行发展


def apply_demo_safety_governor(state: MarsPulseState, actions: np.ndarray) -> np.ndarray:
    """边缘侧强行介入的【工业级安全保护硬围栏】拦截机制：
    不管深度学习 AI 计算出什么动作，如果遭遇极端极端突发情况，该硬编码围栏拥有最高主权强行修改供电动作
    """
    guarded = np.asarray(actions, dtype=np.float32).copy()
    energy_frac = state.energy_kwh / 3000.0
    oxygen_frac = state.oxygen_kg / 500.0

    if state.oxygen_leak:
        guarded[0] = max(guarded[0], 22.0) # 发生漏气故障：制氧供电强制锁死不低于 22 kW
        guarded[1] = min(guarded[1], 6.0)  # 严格限制加热能耗
        guarded[2] = 0.0                   # 彻底切断科学实验供电
    elif state.storm_active or energy_frac < 0.35:
        guarded[0] = min(max(guarded[0], 10.0), 14.0) # 沙尘暴期间将制氧限制在低耗维生状态
        guarded[1] = min(guarded[1], 4.0)
        guarded[2] = 0.0                               # 同样强制熔断科学实验
    elif oxygen_frac < 0.45:
        guarded[0] = max(guarded[0], 20.0) # 常规低氧：制氧强制拉高至 20 kW 以上
        guarded[2] = min(guarded[2], 4.0)  # 科研供电降到极低

    return np.clip(guarded, [0.0, 0.0, 0.0], [30.0, 20.0, 25.0])


def main() -> None:
    """火星脉搏系统主入口：统筹加载硬件、AI权重并运行无限循环控制图"""
    io = UniHikerIO()
    policy = NumpyMAPPOPolicy(POLICY_PATH)
    state = MarsPulseState()

    print("MarsPulse UniHiker M10 landscape demo started.")
    print("Policy loaded:", policy.loaded, POLICY_PATH)
    print("LED strip pin:", LED_PIN_NAME)
    print(f"Current Button Monitoring Port: {EXTERNAL_BUTTON_PIN_NAME} (Pull-Up, Active-Low)")

    tick = 0
    while True:
        # 1. 物理环境传感器实时高速硬采样
        light_value = io.read_light()
        state.light_value = light_value
        # 若光敏读数低于设定阈值，意味着用手把光敏盖住了，立即触发沙尘暴状态
        state.storm_active = light_value < LIGHT_STORM_THRESHOLD
        
        # 实时检测 P23 故障模拟按键状态
        state.oxygen_leak = io.read_fault_button()
        state.button_value = io.last_button_value
        
        # 2. 状态映射逻辑切换
        state.tau = 3.0 if state.storm_active else 0.15 # 沙尘暴时大气光学系数直接拉满到 3.0
        state.failures = [1.0 if state.oxygen_leak else 0.0, 0.0, 0.0, 0.0]
        state.mode_text = "STORM" if state.storm_active else "NORMAL"
        if state.oxygen_leak:
            state.mode_text = "FAULT"

        # 3. 边缘端 AI 智能内核前向推理，求出当前环境的最优资源配置方案
        actions = policy.predict(state)
        # 穿过工业安全强围栏，确保万无一失
        actions = apply_demo_safety_governor(state, actions)
        state.ai_text = describe_ai(state)
        
        # 4. 将 AI 敲定的动作投喂给物理仿真引擎，演进出下一个时刻火星基地的资源指标状态
        update_physics(state, actions)

        # 5. 硬件外设实时物理刷新驱动输出
        io.update_screen(state, tick) # 更新屏幕画面
        io.update_leds(state, tick)   # 更新 WS2812B 渐变流水灯带

        tick += 1
        time.sleep(LOOP_DELAY)        # 主循环精准时间步节拍同步控制


if __name__ == "__main__":
    main()