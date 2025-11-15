import importlib
import pkgutil
from .register import function_register

# 自动导入当前包下的所有子模块
def import_all_modules(package):
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)


# 执行自动导入
import_all_modules(__import__(__name__, fromlist=['']))


# 将注册器全局化
__all__ = ['function_register'] 