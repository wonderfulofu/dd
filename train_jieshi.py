"""
训练脚本：用来自定义数据集训练目标检测模型

这个文件的作用可以一句话概括为：
    负责把 “数据集 + 模型 + 损失函数 + 优化器 + 验证流程” 全部串起来，
    跑完整个训练过程。

使用示例：
    python path/to/train.py --data coco128.yaml --weights yolov3.pt --img 640
"""

# =========================
# 1. 基础库导入
# =========================
import argparse              # 解析命令行参数，例如 --epochs 100
import math                  # 数学函数，比如 log、ceil
import os                    # 操作系统相关，如环境变量、路径
import random                # 随机数
import sys                   # Python 解释器相关，如修改模块搜索路径
import time                  # 计时
from copy import deepcopy    # 深拷贝对象
from datetime import datetime
from pathlib import Path     # 更方便的路径处理

# =========================
# 2. 第三方库导入
# =========================
import numpy as np
import torch
import torch.distributed as dist          # 多卡分布式训练
import torch.nn as nn
import yaml
from torch.cuda import amp                # 自动混合精度训练
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm                     # 进度条

# =========================
# 3. 处理当前文件和工程根目录
# =========================
FILE = Path(__file__).resolve()           # 当前 train.py 的绝对路径
ROOT = FILE.parents[0]                    # 根目录，这里是当前文件所在目录

# 如果工程根目录不在 Python 模块搜索路径里，就加进去
# 这样后面才能正确 import models / utils 等模块
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 转成相对路径，打印日志时更简洁
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# =========================
# 4. 导入工程内部模块
# =========================
import val                                 # 验证脚本，用于每个 epoch 结束后算 mAP
from models.experimental import attempt_load
from models.yolo import Model              # 你的核心模型定义就在 yolo.py
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (
    LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
    check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
    init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
    one_cycle, print_args, print_mutation, strip_optimizer
)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss         # 损失函数类
from utils.metrics import fitness          # 用来综合评价结果优劣
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, select_device,
    torch_distributed_zero_first
)

# =========================
# 5. 分布式训练相关环境变量
# =========================
# 这些变量主要在 DDP（多卡分布式训练）时有用
# 单卡训练通常：
#   LOCAL_RANK = -1
#   RANK = -1
#   WORLD_SIZE = 1
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


# ============================================================
# train()：训练主函数
# 这是整个 train.py 里最核心的函数
# ============================================================
def train(hyp, opt, device, callbacks):
    """
    参数说明：
    hyp: 超参数，可以是 yaml 文件路径，也可以是字典
    opt: 命令行参数对象，包含 epochs、batch-size、imgsz 等
    device: 当前使用的设备，例如 cuda:0 或 cpu
    callbacks: 回调系统，用于日志、保存、可视化等
    """

    # --------------------------------------------------------
    # 1. 从 opt 中取出常用参数，后面书写更方便
    # --------------------------------------------------------
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # --------------------------------------------------------
    # 2. 创建保存目录
    # --------------------------------------------------------
    # 训练时会在 runs/train/exp... 这种目录下保存结果
    # 其中 weights 目录里会放 best.pt 和 last.pt
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # --------------------------------------------------------
    # 3. 读取超参数 hyp
    # --------------------------------------------------------
    # 如果传入的是字符串，就把它当 yaml 文件读取
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    # 打印所有超参数，方便日志记录
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # --------------------------------------------------------
    # 4. 保存本次训练设置
    # --------------------------------------------------------
    # 保存 hyp.yaml：记录本次训练使用的超参数
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    # 保存 opt.yaml：记录本次运行命令行参数
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    data_dict = None

    # --------------------------------------------------------
    # 5. 创建日志系统
    # --------------------------------------------------------
    # 只有主进程（或单卡）才创建 loggers，避免多卡重复打印和重复写日志
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)

        # 如果启用了 wandb，它可能会接管数据集信息
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                # resume 时重新恢复这些参数
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # 把 loggers 中支持的回调函数都注册到 callbacks 中
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # --------------------------------------------------------
    # 6. 基础配置
    # --------------------------------------------------------
    plots = not evolve                # 进化模式时一般不画图
    cuda = device.type != 'cpu'       # 是否用 GPU
    init_seeds(1 + RANK)              # 固定随机种子，保证可复现

    # torch_distributed_zero_first 的作用：
    # 多卡时只让主进程先做某些操作（比如检查数据集），其他进程等它完成
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)

    # 从数据集配置中取训练集/验证集路径
    train_path, val_path = data_dict['train'], data_dict['val']

    # 类别数
    nc = 1 if single_cls else int(data_dict['nc'])

    # 类别名
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']

    # 检查类别名数量和 nc 是否一致
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'

    # 判断是否是 COCO 验证集
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')

    # --------------------------------------------------------
    # 7. 创建模型
    # --------------------------------------------------------
    # 先检查 weights 是否是 .pt
    check_suffix(weights, '.pt')

    # 如果是 .pt 结尾，说明是预训练权重
    pretrained = weights.endswith('.pt')

    if pretrained:
        # ----------------------------
        # 7.1 加载预训练权重
        # ----------------------------
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # 本地没有就尝试下载

        ckpt = torch.load(weights, map_location=device)  # 加载 checkpoint

        # 创建模型：
        # 优先使用传入的 cfg，否则用 checkpoint 里记录的 yaml
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

        # 如果用户改了 cfg 或 anchors，有些参数就不加载
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []

        # checkpoint 的 state_dict
        csd = ckpt['model'].float().state_dict()

        # 只保留和当前模型形状匹配的键
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)

        # 加载到当前模型中
        model.load_state_dict(csd, strict=False)

        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')

    else:
        # ----------------------------
        # 7.2 不使用预训练权重，直接从 yaml 新建模型
        # ----------------------------
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # --------------------------------------------------------
    # 8. 冻结部分层（可选）
    # --------------------------------------------------------
    # freeze=0 表示不冻结
    # freeze=10 表示冻结前 10 层
    freeze = [f'model.{x}.' for x in range(freeze)]

    for k, v in model.named_parameters():
        v.requires_grad = True  # 默认所有参数都训练
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # --------------------------------------------------------
    # 9. 图像尺寸检查
    # --------------------------------------------------------
    # gs = max stride，例如 32
    gs = max(int(model.stride.max()), 32)

    # 检查 imgsz 是否是 stride 的倍数，不是的话自动修正
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # --------------------------------------------------------
    # 10. 自动 batch-size（可选）
    # --------------------------------------------------------
    # 只有单卡且 batch_size=-1 才会自动估计
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz)

    # --------------------------------------------------------
    # 11. 构造优化器
    # --------------------------------------------------------
    nbs = 64  # nominal batch size，参考基准 batch
    accumulate = max(round(nbs / batch_size), 1)  # 梯度累计步数

    # 根据实际 batch 和 accumulate 缩放 weight_decay
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # 参数分组：
    # g0: BN 权重（不加衰减）
    # g1: 普通权重（加 weight decay）
    # g2: bias（通常不加 weight decay）
    g0, g1, g2 = [], [], []

    for v in model.modules():
        # bias
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)

        # 这里特别注意：
        # 代码把 BatchNorm3d 的 weight 放到不衰减组里
        # 因为你的 SNN 代码里 batch_norm_2d 底层就是 BatchNorm3d
        if isinstance(v, nn.BatchNorm3d):
            g0.append(v.weight)

        # 普通 weight
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    # 选择优化器
    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 把另外两组参数加入优化器
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})

    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
        f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias"
    )

    del g0, g1, g2

    # --------------------------------------------------------
    # 12. 学习率调度器
    # --------------------------------------------------------
    if opt.linear_lr:
        # 线性衰减
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        # one_cycle / cosine
        lf = one_cycle(1, hyp['lrf'], epochs)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # --------------------------------------------------------
    # 13. EMA（指数滑动平均）
    # --------------------------------------------------------
    # 用于维护一个“平滑版模型”，验证常用它，通常更稳定
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # --------------------------------------------------------
    # 14. Resume（恢复训练）
    # --------------------------------------------------------
    start_epoch, best_fitness = 0, 0.0

    if pretrained:
        # 恢复优化器状态
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # 恢复 EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # 恢复 epoch
        start_epoch = ckpt['epoch'] + 1

        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'

        # 如果你现在设置的 epochs 比 checkpoint 记录的还小，
        # 代码会自动理解成“继续微调更多轮”
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']

        del ckpt, csd

    # --------------------------------------------------------
    # 15. 多卡 DataParallel（不推荐）
    # --------------------------------------------------------
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            'WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.'
        )
        model = torch.nn.DataParallel(model)

    # --------------------------------------------------------
    # 16. SyncBatchNorm（DDP专用）
    # --------------------------------------------------------
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # --------------------------------------------------------
    # 17. 创建训练集 dataloader
    # --------------------------------------------------------
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,               # 训练时开增强
        cache=opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True
    )

    # 所有标签中出现的最大类别 id
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())

    # 每个 epoch 一共有多少个 batch
    nb = len(train_loader)

    # 检查标签类别 id 是否超出 nc 范围
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # --------------------------------------------------------
    # 18. 创建验证集 dataloader
    # --------------------------------------------------------
    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,               # 验证通常使用矩形加载
            rank=-1,
            workers=workers,
            pad=0.5,
            prefix=colorstr('val: ')
        )[0]

        # 如果不是 resume，则做一些预训练前的检查与可视化
        if not resume:
            labels = np.concatenate(dataset.labels, 0)

            # 画标签分布图
            if plots:
                plot_labels(labels, names, save_dir)

            # 自动 anchor 检查
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

            model.float()  # 预处理后再转回 float

        callbacks.run('on_pretrain_routine_end')

    # --------------------------------------------------------
    # 19. DDP 模式包装模型
    # --------------------------------------------------------
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # --------------------------------------------------------
    # 20. 给模型挂训练相关属性
    # --------------------------------------------------------
    nl = de_parallel(model).model[-1].nl  # 检测层数

    # 根据检测层数、类别数、图像尺寸缩放损失权重
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl

    hyp['label_smoothing'] = opt.label_smoothing

    # 把这些信息挂在 model 上，后面 loss / val / logger 都可能会用
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # --------------------------------------------------------
    # 21. 训练前最后准备
    # --------------------------------------------------------
    t0 = time.time()  # 记录训练开始时间

    # warmup iteration 数
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)

    last_opt_step = -1
    maps = np.zeros(nc)                          # 每个类别的 mAP
    results = (0, 0, 0, 0, 0, 0, 0)             # P, R, mAP50, mAP50-95, val losses...
    scheduler.last_epoch = start_epoch - 1      # 保证 scheduler 从正确 epoch 开始
    scaler = amp.GradScaler(enabled=cuda)       # 混合精度梯度缩放器
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)            # 初始化损失函数

    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )

    # ========================================================
    # 22. epoch 循环：每个 epoch 训练一轮
    # ========================================================
    for epoch in range(start_epoch, epochs):
        model.train()   # 切换到训练模式

        # ----------------------------------------------------
        # 22.1 image weights（可选）
        # ----------------------------------------------------
        # 按类别难度动态调整图像采样概率
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        # ----------------------------------------------------
        # 22.2 初始化本 epoch 的平均 loss
        # ----------------------------------------------------
        mloss = torch.zeros(3, device=device)  # box, obj, cls 三个平均损失

        # DDP 下每个 epoch 要设 sampler 的 epoch，保证 shuffle 正常
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        # pbar 用于进度条显示
        pbar = enumerate(train_loader)

        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

        if RANK in [-1, 0]:
            pbar = tqdm(
                pbar,
                total=nb,
                ncols=NCOLS,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )

        optimizer.zero_grad()

        # ====================================================
        # 23. batch 循环：每次取一个 batch
        # ====================================================
        for i, (imgs, targets, paths, _) in pbar:
            # 从训练开始到现在，总共处理过的 batch 数
            ni = i + nb * epoch

            # 图片搬到设备上，并从 uint8 转成 float32，再归一化到 0~1
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # ------------------------------------------------
            # 23.1 Warmup
            # ------------------------------------------------
            # 前 nw 个 iteration 内动态调整学习率和动量
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())

                for j, x in enumerate(optimizer.param_groups):
                    # 第 3 组参数（bias）有单独的 warmup 起始 lr
                    x['lr'] = np.interp(
                        ni,
                        xi,
                        [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)]
                    )

                    # 如果该参数组里有 momentum，也做线性 warmup
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni,
                            xi,
                            [hyp['warmup_momentum'], hyp['momentum']]
                        )

            # ------------------------------------------------
            # 23.2 Multi-scale（可选）
            # ------------------------------------------------
            # 训练中随机改变输入尺寸，提高鲁棒性
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])  # 缩放因子

                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # ------------------------------------------------
            # 23.3 前向传播
            # ------------------------------------------------
            with amp.autocast(enabled=cuda):
                # 这一句是训练最关键的入口
                # 它会进入 yolo.py 的 Model.forward()
                # 在你的工程里，Model.forward() 会把普通 [B,C,H,W]
                # 复制成 SNN 使用的时间步输入，再跑整个网络
                pred = model(imgs)

                # 根据模型输出和 targets 计算 YOLO 检测损失
                loss, loss_items = compute_loss(pred, targets.to(device))

                # DDP 下 loss 要乘上 WORLD_SIZE
                if RANK != -1:
                    loss *= WORLD_SIZE

                # quad 模式下再额外放大
                if opt.quad:
                    loss *= 4.

            # ------------------------------------------------
            # 23.4 反向传播
            # ------------------------------------------------
            scaler.scale(loss).backward()

            # ------------------------------------------------
            # 23.5 参数更新
            # ------------------------------------------------
            # 只有达到梯度累计步数时，才真正 optimizer.step()
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)   # 等价于 optimizer.step()，但支持混合精度
                scaler.update()
                optimizer.zero_grad()

                # 更新 EMA 模型
                if ema:
                    ema.update(model)

                last_opt_step = ni

            # ------------------------------------------------
            # 23.6 日志显示
            # ------------------------------------------------
            if RANK in [-1, 0]:
                # 更新平均 loss
                mloss = (mloss * i + loss_items) / (i + 1)

                # 当前显存占用
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'

                # 更新进度条显示
                pbar.set_description(
                    ('%10s' * 2 + '%10.4g' * 5) % (
                        f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]
                    )
                )

                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)

        # ====================================================
        # 24. 一个 epoch 结束后
        # ====================================================

        # 更新学习率调度器
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # 只有主进程/单卡做验证和保存
        if RANK in [-1, 0]:
            # 一个 epoch 训练结束的回调
            callbacks.run('on_train_epoch_end', epoch=epoch)

            # 把 model 上的一些属性同步给 ema.ema
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

            # 是否是最后一个 epoch，或者 early stop 即将停止
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            # ------------------------------------------------
            # 24.1 验证集评估
            # ------------------------------------------------
            if not noval or final_epoch:
                results, maps, _ = val.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE,
                    imgsz=imgsz,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss
                )

            # ------------------------------------------------
            # 24.2 更新 best_fitness
            # ------------------------------------------------
            # fitness 是一个综合评价指标，通常综合了 P / R / mAP
            fi = fitness(np.array(results).reshape(1, -1))

            if fi > best_fitness:
                best_fitness = fi

            # 记录本轮训练+验证的日志值
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # ------------------------------------------------
            # 24.3 保存模型
            # ------------------------------------------------
            if (not nosave) or (final_epoch and not evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)),
                    'ema': deepcopy(ema.ema),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()
                }

                # 保存 last.pt
                torch.save(ckpt, last)

                # 如果当前最好，则保存 best.pt
                if best_fitness == fi:
                    torch.save(ckpt, best)

                # 如果设置了 save_period，则周期性多存一份
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')

                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # ------------------------------------------------
            # 24.4 单卡 EarlyStopping
            # ------------------------------------------------
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

    # ========================================================
    # 25. 训练全部结束后
    # ========================================================
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        # 对 last.pt 和 best.pt 进一步处理
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 去掉优化器等无关信息，减小体积

                # 对 best.pt 再做一次正式验证
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE,
                        imgsz=imgsz,
                        model=attempt_load(f, device),
                        iou_thres=0.65 if is_coco else 0.60,
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=True,
                        callbacks=callbacks,
                        compute_loss=compute_loss
                    )

                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    # 清空显存缓存
    torch.cuda.empty_cache()

    return results


# ============================================================
# parse_opt()：定义命令行参数
# ============================================================
def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    # 预训练权重路径
    parser.add_argument('--weights', type=str, default=ROOT / 'runs/train/exp10/weights/best.pt',
                        help='initial weights path')

    # 模型结构 yaml
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/resnet10.yaml',
                        help='model.yaml path')

    # 数据集配置 yaml
    parser.add_argument('--data', type=str, default=ROOT / 'datasets/banana/data.yaml',
                        help='dataset.yaml path')

    # 超参数 yaml
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml',
                        help='hyperparameters path')

    # 训练 epoch 数
    parser.add_argument('--epochs', type=int, default=20)

    # batch 大小，-1 表示自动估计
    parser.add_argument('--batch-size', type=int, default=4,
                        help='total batch size for all GPUs, -1 for autobatch')

    # 训练和验证图片尺寸
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                        help='train, val image size (pixels)')

    # 是否使用矩形训练
    parser.add_argument('--rect', action='store_true', help='rectangular training')

    # 是否从最近一次中断继续训练
    parser.add_argument('--resume', nargs='?', const=False, default=False,
                        help='resume most recent training')

    # 只保存最终模型
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')

    # 只验证最后一个 epoch
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')

    # 关闭自动 anchor 检查
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')

    # 超参数进化
    parser.add_argument('--evolve', type=int, nargs='?', const=300,
                        help='evolve hyperparameters for x generations')

    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')

    # 图片缓存到内存/磁盘
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                        help='--cache images in "ram" (default) or "disk"')

    # 是否使用 image_weights 采样策略
    parser.add_argument('--image-weights', action='store_true',
                        help='use weighted image selection for training')

    # 设备
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3,4,5,6,7 or cpu')

    # 多尺度训练
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')

    # 多类当单类训练
    parser.add_argument('--single-cls', action='store_true',
                        help='train multi-class data as single-class')

    # 使用 Adam
    parser.add_argument('--adam', action='store_true',
                        help='use torch.optim.Adam() optimizer')

    # DDP 下使用 SyncBN
    parser.add_argument('--sync-bn', action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')

    # dataloader worker 数
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')

    # 项目保存目录
    parser.add_argument('--project', default=ROOT / 'runs/train',
                        help='save to project/name')

    # 实验名称
    parser.add_argument('--name', default='exp',
                        help='save to project/name')

    # 已有目录是否允许直接使用
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')

    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')

    # 标签平滑
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon')

    # EarlyStopping 容忍轮数
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience (epochs without improvement)')

    # 冻结层数
    parser.add_argument('--freeze', type=int, default=0,
                        help='Number of layers to freeze. backbone=10, all=24')

    # 每隔多少 epoch 额外保存一次 checkpoint
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs (disabled if < 1)')

    # DDP 参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DDP parameter, do not modify')

    # -------------------------
    # Weights & Biases 相关参数
    # -------------------------
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest',
                        help='W&B: Version of dataset artifact to use')

    # known=True 时只解析已知参数，常用于其他脚本调用
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


# ============================================================
# main()：脚本主入口
# ============================================================
def main(opt, callbacks=Callbacks()):
    # --------------------------------------------------------
    # 1. 运行前检查
    # --------------------------------------------------------
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)     # 打印所有参数
        check_git_status()             # 检查 git 状态
        check_requirements(exclude=['thop'])  # 检查依赖库

    # --------------------------------------------------------
    # 2. Resume 逻辑
    # --------------------------------------------------------
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:
        # 如果指定 resume，就从最近中断的 checkpoint 继续
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'

        # 读取上次训练保存的 opt.yaml
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))

        # 恢复 checkpoint 路径
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        LOGGER.info(f'Resuming training from {ckpt}')

    else:
        # 正常训练时，检查 data/cfg/hyp 文件路径
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)

        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

        # 如果是超参数进化，保存目录切到 runs/evolve
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False

        # 生成本次实验保存目录，如 runs/train/exp, exp2...
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # --------------------------------------------------------
    # 3. 设备选择与 DDP 初始化
    # --------------------------------------------------------
    device = select_device(opt.device, batch_size=opt.batch_size)

    if LOCAL_RANK != -1:
        # DDP 条件检查
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'

        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)

        # 初始化进程组
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # --------------------------------------------------------
    # 4. 正常训练
    # --------------------------------------------------------
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

        # 多卡训练结束后销毁进程组
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # --------------------------------------------------------
    # 5. 超参数进化（可选）
    # --------------------------------------------------------
    else:
        # meta 定义了每个可进化超参数的：
        # (是否参与变异, 下界, 上界)
        meta = {
            'lr0': (1, 1e-5, 1e-1),
            'lrf': (1, 0.01, 1.0),
            'momentum': (0.3, 0.6, 0.98),
            'weight_decay': (1, 0.0, 0.001),
            'warmup_epochs': (1, 0.0, 5.0),
            'warmup_momentum': (1, 0.0, 0.95),
            'warmup_bias_lr': (1, 0.0, 0.2),
            'box': (1, 0.02, 0.2),
            'cls': (1, 0.2, 4.0),
            'cls_pw': (1, 0.5, 2.0),
            'obj': (1, 0.2, 4.0),
            'obj_pw': (1, 0.5, 2.0),
            'iou_t': (0, 0.1, 0.7),
            'anchor_t': (1, 2.0, 8.0),
            'anchors': (2, 2.0, 10.0),
            'fl_gamma': (0, 0.0, 2.0),
            'hsv_h': (1, 0.0, 0.1),
            'hsv_s': (1, 0.0, 0.9),
            'hsv_v': (1, 0.0, 0.9),
            'degrees': (1, 0.0, 45.0),
            'translate': (1, 0.0, 0.9),
            'scale': (1, 0.0, 0.9),
            'shear': (1, 0.0, 10.0),
            'perspective': (0, 0.0, 0.001),
            'flipud': (1, 0.0, 1.0),
            'fliplr': (0, 0.0, 1.0),
            'mosaic': (1, 0.0, 1.0),
            'mixup': (1, 0.0, 1.0),
            'copy_paste': (1, 0.0, 1.0)
        }

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            if 'anchors' not in hyp:
                hyp['anchors'] = 3

        # 进化时通常只验证最终结果，不保存每一轮训练过程
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)

        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'

        # 如果指定 bucket，就从云端下载 evolve.csv
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')

        # 进行多代进化
        for _ in range(opt.evolve):
            if evolve_csv.exists():
                # ------------------------
                # 5.1 选择父代
                # ------------------------
                parent = 'single'  # 单个父代 or 加权父代
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)

                n = min(5, len(x))  # 只考虑前 n 个最优结果
                x = x[np.argsort(-fitness(x))][:n]

                w = fitness(x) - fitness(x).min() + 1E-6

                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                # ------------------------
                # 5.2 变异
                # ------------------------
                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])
                ng = len(meta)
                v = np.ones(ng)

                # 保证至少发生一次变化
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)

                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            # ------------------------
            # 5.3 限制超参数范围
            # ------------------------
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])   # 下界
                hyp[k] = min(hyp[k], v[2])   # 上界
                hyp[k] = round(hyp[k], 5)

            # ------------------------
            # 5.4 用当前变异后的超参数训练一次
            # ------------------------
            results = train(hyp.copy(), opt, device, callbacks)

            # 记录结果
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # 画进化结果图
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}'
        )


# ============================================================
# run()：提供给其他脚本调用的接口
# ============================================================
def run(**kwargs):
    """
    用法示例：
        import train
        train.run(data='coco128.yaml', imgsz=320, weights='yolov3.pt')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


# ============================================================
# 脚本入口
# ============================================================
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)