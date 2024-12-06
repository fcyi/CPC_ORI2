"""
Sacred experiment file
"""

# Sacred被广泛应用于深度学习和机器学习领域，特别是在PyTorch和TensorFlow等框架中。
# 它也是许多研究实验的标准工具之一，已经被许多知名的机器学习项目所采用。
# sacred就是一种和wandb，tensorboard并列的实验结果的工具
# 使用Sacred，我们可以方便地记录实验元数据和结果，优化超参数和运行并行实验，并可视化和分析实验结果。
# Sacred还支持与其他Python库和优化库集成，以满足不同的实验需求和工作流程。
# 它的主要功能包括：
#
#     实验管理：记录实验运行的元数据和结果，方便回溯和复现。
#     超参数调整：支持定义超参数范围、采样策略和搜索算法，并可以与其他优化库集成。
#     并行运行：支持在本地和远程计算机上并行运行实验，并可以在多个进程和线程中共享资源。
#     可视化和分析：支持将实验结果可视化和分析，以便更好地理解和优化实验结果。

# Sacred
from sacred import Experiment  # Experiment: Sacred 中的核心类，用于创建实验对象。你可以通过它来配置实验、记录日志、存储结果等。
from sacred.stflow import LogFileWriter  # LogFileWriter: 用于将日志写入文件。虽然代码中并未直接使用，但它通常是与 SACRED 配合来记录日志。
# FileStorageObserver用来将实验的所有数据（比如配置、结果、日志等）保存在本地文件中。
# 用于将实验数据存储到 MongoDB 数据库中。虽然在此代码中未使用，但它通常与 FileStorageObserver 配合。
from sacred.observers import FileStorageObserver, MongoObserver

# custom config hook
from utils.yaml_config_hook import yaml_config_hook

# 创建了一个名为“my_experiment”的Sacred实验
# ex 对象用于定义实验配置、执行函数和记录实验的所有相关数据。
ex = Experiment("contrastive-predictive-coding")

#### file output directory
# FileStorageObserver 是一个 Observer，用于将实验数据存储到文件系统中。
# 这里指定了一个存储目录 ./logs。
# 通过 ex.observers.append(...) 方法，
# 将该观察者添加到实验对象中，所有的实验日志、配置等都会保存到这个目录中。
ex.observers.append(FileStorageObserver("./logs"))

#### database output
# ex.observers.append(
#     MongoObserver().create(
#         url=f"mongodb://admin:admin@localhost:27017/?authMechanism=SCRAM-SHA-1",
#         db_name="db",
#     )
# )


# 然后定义了一个配置函数my_config()。
@ex.config
def my_config():
    yaml_config_hook("./config/audio/config.yaml", ex)

    # override any settings here
    # start_epoch = 100
    # ex.add_config(
    #   {'start_epoch': start_epoch})
