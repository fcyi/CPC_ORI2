import os
import argparse
import time
import torch
import numpy as np
from datetime import datetime

# Apex for mixed-precision training
from apex import amp


# TensorBoard
from torch.utils.tensorboard import SummaryWriter


from model import load_model, save_model
from data.loaders import librispeech_loader
from validation import validate_speakers

#### pass configuration
from experiment import ex


def train(args, model, optimizer, writer):

    # get datasets and dataloaders
    (train_loader, train_dataset, test_loader, test_dataset,) = librispeech_loader(
        args, num_workers=args.num_workers
    )

    total_step = len(train_loader)
    print_idx = 100

    # at which step to validate training
    validation_idx = 1000

    best_loss = 0

    start_time = time.time()
    global_step = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        loss_epoch = 0
        for step, (audio, filename, _, start_idx) in enumerate(train_loader):

            start_time = time.time()

            if step % validation_idx == 0:
                validate_speakers(args, train_dataset, model, optimizer, epoch, step, global_step, writer)

            audio = audio.to(args.device)

            # forward
            loss = model(audio)

            # accumulate losses for all GPUs
            loss = loss.mean()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # backward, depending on mixed-precision
            # model.zero_grad() 和 optimizer.zero_grad() 在深度学习框架（例如 PyTorch）中用于清除梯度，但它们的作用范围和具体实现有所不同。
            # model.zero_grad()
            # 功能: model.zero_grad() 是在模型（如神经网络）上调用的，它会将 model 中所有参数的梯度置为零。
            # 作用范围: 这个方法会遍历模型中的所有参数（通常是 torch.nn.Module 的子类，这是个问题，因为有些模型参数可能是根据nn.Parameters定义的，而并非属于nn.Module类，
            #          torch.nn.Parameter 是 PyTorch 中用于表示模型参数的一个类。它是一个张量（tensor），并且在被加入到 nn.Module 时，会自动被注册为该模块的可学习参数），
            #          并将每个参数的 .grad 属性设置为 None 或零。
            # 使用场景: 通常在进行一次训练迭代之前调用，以确保在执行反向传播时不会累加先前迭代的梯度。
            #
            # optimizer.zero_grad()
            # 功能: optimizer.zero_grad() 是在优化器对象上调用的，它的作用也是清零与优化器关联的所有参数的梯度。
            # 作用范围: 这个方法只会清除优化器所管理的参数的梯度，通常是那些在创建优化器时指定的参数。
            #          比如，如果你只为部分模型参数创建了优化器，那么调用 optimizer.zero_grad() 只会清除这些参数的梯度。
            # 使用场景: 同样，在每次调用 optimizer.step()（更新参数之前）之前，通常需要调用这个方法来重置梯度。
            model.zero_grad()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if step % print_idx == 0:
                examples_per_second = args.batch_size / (time.time() - start_time)
                print(
                    "[Epoch {}/{}] Train step {:04d}/{:04d} \t Examples/s = {:.2f} \t "
                    "Loss = {:.4f} \t Time/step = {:.4f}".format(
                        epoch,
                        args.num_epochs,
                        step,
                        len(train_loader),
                        examples_per_second,
                        loss,
                        time.time() - start_time,
                    )
                )

            writer.add_scalar("Loss/train_step", loss, global_step)
            loss_epoch += loss
            global_step += 1

        avg_loss = loss_epoch / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        # 使用ex.log_scalar()函数记录了实验结果的准确度。
        ex.log_scalar("loss.train", avg_loss, epoch)

        conv = 0
        for idx, layer in enumerate(model.module.model.modules()):
            if isinstance(layer, torch.nn.Conv1d):
                writer.add_histogram(
                    "Conv/weights-{}".format(conv),
                    layer.weight,
                    global_step=global_step,
                )
                conv += 1

            if isinstance(layer, torch.nn.GRU):
                writer.add_histogram(
                    "GRU/weight_ih_l0", layer.weight_ih_l0, global_step=global_step
                )
                writer.add_histogram(
                    "GRU/weight_hh_l0", layer.weight_hh_l0, global_step=global_step
                )

        if avg_loss > best_loss:
            best_loss = avg_loss
            save_model(args, model, optimizer, best=True)

        # save current model state
        save_model(args, model, optimizer)
        args.current_epoch += 1


# 接下来，我们定义了一个main()函数作为实验的主体部分，
# 并使用Sacred装饰器@ex.automain将其指定为自动运行的函数。
@ex.automain
def main(_run, _log):
    # 利用这招可以将config文件夹下的所有的.yaml文件都整合成一个字典
    # _run: 通常是一个对象，包含与当前实验运行相关的信息，包括配置、状态等。
    # _log: 通常用于记录日志信息，帮助用户跟踪实验的执行过程。
    #  _run.config 代表当前实验运行的配置，通常是一个字典形式。
    #  这里的配置可能来自于多个 YAML 配置文件的合并结果。
    #  argparse.Namespace(**_run.config) 将这个配置字典解包为命名空间的形式，
    #  使得可以通过属性访问的方式来使用这些配置值，例如：args.some_config_value。
    args = argparse.Namespace(**_run.config)

    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    else:
        out_dir = _run.observers[0].dir

    args.out_dir = out_dir

    # set start time
    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.current_epoch = args.start_epoch

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model
    model, optimizer = load_model(args)

    # initialize TensorBoard
    tb_dir = os.path.join(out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    # writer.add_graph(model.module, torch.rand(args.batch_size, 1, 20480).to(args.device))

    try:
        train(args, model, optimizer, writer)
    except KeyboardInterrupt:
        print("Interrupting training, saving model")

    save_model(args, model, optimizer)


if __name__ == "__main__":
    main()
