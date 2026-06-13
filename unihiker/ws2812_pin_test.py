"""
MarsPulse — WS2812B LED灯带测试脚本
===================================
模块说明: 行空板WS2812B灯带引脚连通性测试。
         循环显示红/绿/蓝/白四色，验证硬件连接正常。
硬件: UniHiker, WS2812B LED灯带
作者: 陈凯熙
开发时间: 2026年5月
"""

# -*- coding: utf-8 -*-
"""Quick WS2812 pin test for UniHiker M10.

Watch the LED strip while this runs. It tests P21, P24 and P2 in sequence.
If one section lights, use that pin as MARSPULSE_LED_PIN for the main demo.
"""

import time

from pinpong.board import Board, Pin, NeoPixel
import pinpong.board as pb


TEST_PINS = ["P21"]
PIXELS = 7


def set_led(strip, index, color):
    strip[index] = color


def color_hex(color):
    r, g, b = color
    return (r << 16) | (g << 8) | b


def main():
    Board("UNIHIKER").begin()
    for pin_name in TEST_PINS:
        pin_no = getattr(Pin, pin_name)
        strip = pb.NeoPixel(Pin(pin_no), PIXELS)
        strip.brightness(120)
        print("Testing", pin_name)
        strip.clear()
        strip.range_color(0, PIXELS // 3, color_hex((0, 120, 20)))
        strip.range_color(PIXELS // 3, PIXELS * 2 // 3, color_hex((180, 30, 0)))
        strip.range_color(PIXELS * 2 // 3, PIXELS, color_hex((0, 80, 220)))
        time.sleep(5)
        strip.clear()
        time.sleep(1)
    print("Done. If nothing lit, check 5V, GND, DIN direction and common ground.")


if __name__ == "__main__":
    main()
