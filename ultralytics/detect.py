import numpy as np
"""
接口规范模板代码
统一推理算法的接口规范是为了后续使用程序进行统一评估。
训练完模型之后，请在嵌入式设备中实现detect.py的两个函数init getPredict。
在init函数中完成算法模型的所有初始化设置。
getPredict是算法处理的核心逻辑，输入一张图片，输出检测或识别的结果。
detect.py及其依赖模块(参赛者新安装的)、模型以及配置文件，请放在一个文件夹pythonModule中。
确保在文件夹中可正常运行detect.py。
"""
def getPredict(img):
    """
    输入一张图片，输出检测/识别结果 dets

    输入：
    img: numpy.ndarray
        三维数组（高度、宽度、通道数），数据类型为 uint8，通道顺序为 BGR（OpenCV 格式）

    输出：
    dets: numpy.ndarray
        形状为 (目标数量, 7) 的数组，每一行为一个目标的信息：
        [cls, leftx, topy, width, height, conf, reserve]

    字段说明：
    cls: 类别 ID（例如0, 1, 2 如果是检测请输出0）
    leftx: 检测/识别框左上顶点 x 坐标（像素）
    topy: 检测/识别框左上顶点 y 坐标（像素）
    width: 检测/识别框宽度（像素）
    height: 检测/识别框高度（像素）
    conf: 置信度，转化为整数（例如 0.5 对应转化为int(0.5*65535.0)=32767）
    reserve: 保留字，始终设为 0
    """
    dets = []
    dets.append([0, 50, 50, 50, 50, 90, 0])
    dets.append([1, 60, 60, 50, 50, 100, 0])
    dets = np.array(dets)
    return dets


def init(config):
    """
    从配置文件中读取模型初始化的配置文件，并完成算法的初始化

    输入：
    config: str
        配置文件的路径

    输出：
    result: bool
        成功返回 True，失败返回 False
    """
    result = True
    return result

def test():
    """简单测试：初始化 + 推理"""
    # 1. 假装有一张 480×640 的 BGR 图片
    fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 2. 初始化模型
    ok = init("fake_config.txt")
    print("init 返回:", ok)

    # 3. 推理
    dets = getPredict(fake_img)
    print("检测结果 (shape={}):".format(dets.shape))
    print(dets)

if __name__ == "__main__":
    test()
