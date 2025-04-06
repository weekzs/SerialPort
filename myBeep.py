import winsound


def beep_sound(frequency=2000, duration=1000):
    """
    发出指定频率和持续时间的蜂鸣声。

    :param frequency: 声音的频率，单位为赫兹，默认值为 2000
    :param duration: 声音的持续时间，单位为毫秒，默认值为 1000
    """
    winsound.Beep(frequency, duration)

