import os
import yaml


def yaml_config_hook(config_file, ex):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    此处的配置写法和config目录的构建直接相关，因此后续写配置时要十分注意
    """

    ex.add_config(config_file)  # 将主配置文件添加到 ex 对象中。

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)  # 安全地加载 YAML 文件的内容到字典 cfg 中。
        for d in cfg.get("defaults", []):  # 迭代 defaults 字段中的所有条目（即键值对）。如果没有找到 defaults 字段，则使用空列表。
            config_dir, cf = d.popitem()  # 获取字典中的第一个项，这里假设 d 是一个字典，其中包含要加载的其他 YAML 文件的目录和文件名。
            # 获取字典中的第一个项，这里假设 d 是一个字典，其中包含要加载的其他 YAML 文件的目录和文件名。
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                cfg.update(yaml.safe_load(f))  # 将子配置文件的内容合并到主配置字典 cfg 中。

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    ex.add_config(cfg)
    del f
    del cfg
