import logging
import os

from concurrent_log_handler import ConcurrentRotatingFileHandler

debug_log_folder = './logs'  # 定义日志文件夹路径
if not os.path.exists(debug_log_folder):  # 确保日志文件夹存在
    os.makedirs(debug_log_folder)

debug_logger = logging.getLogger('debug_logger')
debug_logger.setLevel(logging.DEBUG)

debug_logger.propagate = False  # 设置 logger 不向上传递日志信息，避免输出到控制台

debug_handler = ConcurrentRotatingFileHandler(os.path.join(debug_log_folder, "debug.log"), "a", 160 * 1024 * 1024, 5)

# 创建一个带有自定义字段的格式器
formatter = logging.Formatter(
    f"%(levelname)s %(asctime)s [PID:%(process)d] %(filename)s:%(funcName)s:%(lineno)d %(message)s")
debug_handler.setFormatter(formatter)  # 设置日志格式

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
debug_logger.addHandler(debug_handler)
