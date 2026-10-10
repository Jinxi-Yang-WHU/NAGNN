class logger(object):
    """
    一个伪文件对象，可以将输出同时分发到多个流（例如屏幕和文件）。
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        """将字符串写入所有已注册的文件流。"""
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保内容被立即写入，而不是等待缓冲区满

    def flush(self):
        """手动刷新所有文件流的缓冲区。"""
        for f in self.files:
            f.flush()