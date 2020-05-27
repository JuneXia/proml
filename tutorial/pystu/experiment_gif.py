"""
opencv python 之 003 录屏并保存为Gif图片
By Linyoubiao
2020-03-19
"""
# from PIL import ImageGrab
import numpy as np
import cv2 as cv
import imageio
import time

if __name__ == "__main__":
    cv.namedWindow("grab", cv.WINDOW_NORMAL)
    buff = []
    size = (0, 0, 500, 500)
    # 获得当前屏幕
    # p = ImageGrab.grab(size)
    # 获得当前屏幕的大小
    # x, y = p.size
    # 编码格式
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # 输出文件
    # video = cv.VideoWriter('d:/test.avi', fourcc, 16, (x, y))
    while True:
        # im = ImageGrab.grab(size)
        img = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
        # video.write(img)
        cv.imshow("grab", img)
        buff.append(img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # 保存图片，时间间隔为0.1s
    gif = imageio.mimsave('./screen.gif', buff, 'GIF', duration=0.1)
    # video.release()
    cv.destroyAllWindows()