import argparse
import copy
import logging
import math
import os
import time
from torch import nn
from external_models.RCR.util import confidence_update, DATASET_GETTERS
from models.PreResNet import PreResNet18
from train_utils import get_cosine_schedule_with_warmup, set_seed, interleave, de_interleave, save_checkpoint
from utils.EarlyStopping import EarlyStopping
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, accuracy
import nni

logger = logging.getLogger(__name__)
best_acc = 0
best_test_acc = 0


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-gpu_id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-num_workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument("-expand_labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('-arch', default='presnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('-total_steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('-eval_step', default=156, type=int,
                        help='number of eval steps to run')
    parser.add_argument('-start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-batch_size', default=256, type=int,
                        help='train batch size')
    parser.add_argument('-lr', default=0.05, type=float,
                        help='initial learning rate')
    parser.add_argument('-warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('-wdecay', default=1e-3, type=float,
                        help='weight decay')
    parser.add_argument('-nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('-use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('-ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('-mu', default=3, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('-lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('-T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('-threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('-out', default='./log',
                        help='directory to output the result')
    parser.add_argument('-resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-seed', default=0, type=int,
                        help="random seed")
    parser.add_argument("-amp", action="store_true", default=False,
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("-exclusion_mode", default='adaptive', choices=['adaptive', 'fix', 'known'],
                        help="is or not fix the exclusion rate.")
    parser.add_argument("-fix_exclusion_rate", default=0.4, type=float,
                        help="fixed exclusion rate.")
    parser.add_argument('-select_ratio', help='select_ratio', type=float, default=2)
    parser.add_argument("-opt_level", type=str, default="O2",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("-local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('-no_progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('-partial', default=True, type=bool,
                        help='use partial label dataset')
    parser.add_argument('-eid', help='experiment_id', type=int, default=1)
    parser.add_argument('-select_learning_rate', type=float, default=1e-1,
                        help='optimizer\'s learning rate')
    parser.add_argument('-select_weight_decay', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('-select_batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('-select_epoch', type=int, default=5,
                        help='max numbers of epochs')
    parser.add_argument('-dataset', type=str, default='cifar10', required=False,
                        choices=['mnist', 'fashion', 'kmnist', 'cifar10', 'cifar100'],
                        help='specify a dataset')
    parser.add_argument('-select_model', type=str, default='mlp', required=False,
                        help='select model name',
                        choices=['linear', 'mlp', 'lenet', 'resnet', 'resnet50', 'densenet'])
    parser.add_argument('-select_scheduler', type=list, default=[200, 400], required=False,
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1")
    parser.add_argument('-select_loss', required=False, default='cce', type=str,
                        help='specify a loss function',
                        choices=['mae', 'mse', 'cce', 'gce', 'phuber_ce', 'fl', 'jinb', 'courb', 'jinu', 'couru'])
    parser.add_argument('-partial_type', type=str, default='noise+partial',
                        choices=['uset', 'ulabel', 'ccnlabel5', 'noise+partial', 'partial+noise'],
                        help='partial dataset type')
    parser.add_argument('-partial_rate', type=float, default=0.3,
                        help='partial rate')
    parser.add_argument('-noisy_rate', type=float, default=0.3,
                        help='noisy rate')
    parser.add_argument('-warmupPRODEN', type=int, default=0,
                        help='epoch for warm up of PRODEN')
    parser.add_argument('-early_stopping', type=bool, default=True,
                        help='use early stopping to stop process.')
    parser.add_argument('-loss', type=str, default='RCR',
                        choices=['RCR'],
                        help='use noisy samples progressive exclusion.')
    parser.add_argument('-exclusion', type=str, default='none',
                        choices=['progressive_exclusion', 'small_loss', 'none'],
                        help='use noisy samples progressive exclusion.')
    parser.add_argument('-exclusion_patience', type=int, default=5,
                        help='exclusion early stopping patience.')
    parser.add_argument('-noise_type', type=str, default='symmetric',
                        choices=['symmetric', 'pairflip'],
                        help='noise type')
    parser.add_argument('-exclude_rate', type=float, default=0.03,
                        help='exclusion rate every epoch.')
    parser.add_argument('-visdom', type=bool, default=False,
                        help='if use visdom to visualize the training process.')
    parser.add_argument('-NNI', action='store_true', default=False,
                        help='use NNI tuning.')

    #parameters for RCR
    parser.add_argument('-lam', default=1, type=float)

    args = parser.parse_args()
    global best_acc, best_test_acc

    if args.NNI:
        params = {
            'select_learning_rate': args.select_learning_rate,
            'select_epoch': args.select_epoch,
            'select_weight_decay': args.select_weight_decay,
            'exclusion_patience': args.exclusion_patience,
            'exclude_rate': args.exclude_rate,
        }
        optimized_params = nni.get_next_parameter()
        params.update(optimized_params)
        args.select_learning_rate = params['select_learning_rate']
        args.select_epoch = params['select_epoch']
        args.select_weight_decay = params['select_weight_decay']
        args.exclusion_patience = params['exclusion_patience']
        args.exclude_rate = params['exclude_rate']

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'presnet':
            import models.PreResNet as models
            model = models.PreResNet18(num_class=args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(os.path.join(args.out,
                                                 'eid_{}pr_{}nr_{}/'.format(args.eid, args.partial_rate,
                                                                            args.noisy_rate)))

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset, valid_dataset = DATASET_GETTERS[args.dataset](args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    if args.exclusion == 'progressive_exclusion':
        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.batch_size * args.mu,
            num_workers=args.num_workers,
            drop_last=True)
    else:
        unlabeled_trainloader = None

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.partial:
        valid_loader = DataLoader(
            valid_dataset,
            sampler=SequentialSampler(valid_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, valid_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, valid_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp

    if args.early_stopping:
        earlyStopping = EarlyStopping(patience=25,
                                      verbose=True,
                                      delta=0,
                                      trace_func=logger.info)

    global best_acc, best_test_acc
    val_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        if args.exclusion == 'progressive_exclusion':
            unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    if args.exclusion == 'progressive_exclusion':
        unlabeled_iter = iter(unlabeled_trainloader)

    # criterion for RCR
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()

    # init confidence for RCR
    labeled_trainloader.dataset.partial_labels = (labeled_trainloader.dataset.partial_labels > 1e-4).float()
    confidence = copy.deepcopy(labeled_trainloader.dataset.partial_labels.numpy())
    confidence = confidence / confidence.sum(axis=1)[:, None]

    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
            losses_u = AverageMeter()
            mask_probs = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        # train one epoch
        for batch_idx in range(args.eval_step):
            try:
                x_aug0, x_aug1, x_aug2, targets_x, indexes_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                x_aug0, x_aug1, x_aug2, targets_x, indexes_x = labeled_iter.next()

            if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
                try:
                    (inputs_u_w, inputs_u_s), _, _ = unlabeled_iter.next()
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), _, _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = x_aug0.shape[0]
            if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
                inputs = interleave(
                    torch.cat((x_aug0, x_aug1, x_aug2, inputs_u_w, inputs_u_s)), 2 * args.mu + 3).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = model(inputs)
                logits = de_interleave(logits, 2 * args.mu + 3)
                y_pred_aug0 = logits[:batch_size]
                y_pred_aug1 = logits[batch_size: 2 * batch_size]
                y_pred_aug2 = logits[2 * batch_size: 3 * batch_size]
                logits_u_w, logits_u_s = logits[3 * batch_size:].chunk(2)
                del logits
            else:
                x_aug0 = x_aug0.to(args.device)
                x_aug1 = x_aug1.to(args.device)
                x_aug2 = x_aug2.to(args.device)
                targets_x = targets_x.to(args.device)
                y_pred_aug0 = model(x_aug0)
                y_pred_aug1 = model(x_aug1)
                y_pred_aug2 = model(x_aug2)

            y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0, dim=-1)
            y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
            y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)

            y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
            y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
            y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

            if args.partial:
                if args.loss == 'RCR':
                    # consist loss
                    consist_loss0 = consistency_criterion(y_pred_aug0_probas_log,
                                                          torch.tensor(confidence[indexes_x]).float().to(args.device))
                    consist_loss1 = consistency_criterion(y_pred_aug1_probas_log,
                                                          torch.tensor(confidence[indexes_x]).float().to(args.device))
                    consist_loss2 = consistency_criterion(y_pred_aug2_probas_log,
                                                          torch.tensor(confidence[indexes_x]).float().to(args.device))
                    # supervised loss
                    super_loss = -torch.mean(
                        torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0, dim=1)) * (1 - targets_x), dim=1))
                    # dynamic lam
                    lam = min((epoch / 100) * args.lam, args.lam)

                    # Unified loss
                    Lx = lam * (consist_loss0 + consist_loss1 + consist_loss2) + super_loss
            else:
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()

                loss = Lx + args.lambda_u * Lu
            else:
                loss = Lx

            # amp control
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
                losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()

            # update confidence
            confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, targets_x, indexes_x, num_classes=args.num_classes)

            # ema control
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
                mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion' else 0,
                        mask=mask_probs.avg if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion' else 0))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, val_loss, test_acc, val_acc = test(args, test_loader, valid_loader, test_model)
            if args.NNI:
                nni.report_intermediate_result(test_acc)
            train_mark = 'eid: {}, pr: {}, nr: {}. '.format(args.eid,
                                                            args.partial_rate,
                                                            args.noisy_rate)
            args.writer.add_scalar(train_mark + '/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar(train_mark + '/2.train_loss_x', losses_x.avg, epoch)
            if args.warmupPRODEN <= epoch and args.exclusion == 'progressive_exclusion':
                args.writer.add_scalar(train_mark + '/3.train_loss_u', losses_u.avg, epoch)
                args.writer.add_scalar(train_mark + '/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar(train_mark + '/5.test_acc', test_acc, epoch)
            args.writer.add_scalar(train_mark + '/6.test_loss', test_loss, epoch)
            args.writer.add_scalar(train_mark + '/7.val_acc', val_acc, epoch)
            args.writer.add_scalar(train_mark + '/8.val_loss', val_loss, epoch)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if is_best:
                # best_test_acc = max(test_acc, best_test_acc)
                best_test_acc = test_acc

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            model_save_path = os.path.join(args.out,
                                           'eid_{}pr_{}nr_{}/'.format(args.eid, args.partial_rate, args.noisy_rate))
            os.makedirs(model_save_path, exist_ok=True)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, model_save_path)

            val_accs.append(val_acc)
            logger.info('Best val top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Best test top-1 acc: {:.2f}'.format(best_test_acc))
            logger.info('Mean val top-1 acc: {:.2f}\n'.format(
                np.mean(val_accs[-20:])))

            # early stopping
            if args.early_stopping:
                earlyStopping(val_acc)
                if earlyStopping.early_stop:
                    logger.info("Training stopped.")
                    if args.NNI:
                        nni.report_final_result(best_test_acc)
                    break

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, valid_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    test_losses = AverageMeter()
    val_losses = AverageMeter()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        if not args.no_progress:
            test_loader = tqdm(test_loader,
                               disable=args.local_rank not in [-1, 0])
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            test_losses.update(loss.item(), inputs.shape[0])
            test_top1.update(prec1.item(), inputs.shape[0])
            test_top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {test_top1:.2f}. top5: {test_top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=test_losses.avg,
                        test_top1=test_top1.avg,
                        test_top5=test_top5.avg,
                    ))

        if not args.no_progress:
            valid_loader = tqdm(valid_loader,
                                disable=args.local_rank not in [-1, 0])
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            val_losses.update(loss.item(), inputs.shape[0])
            val_top1.update(prec1.item(), inputs.shape[0])
            val_top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                valid_loader.set_description(
                    "Valid Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {val_top1:.2f}. top5: {val_top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(valid_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=val_losses.avg,
                        val_top1=val_top1.avg,
                        val_top5=val_top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()
            valid_loader.close()

    logger.info("valid_top-1 acc: {:.2f}".format(val_top1.avg))
    logger.info("valid_top-5 acc: {:.2f}".format(val_top5.avg))
    logger.info("test_top-1 acc: {:.2f}".format(test_top1.avg))
    logger.info("test_top-5 acc: {:.2f}".format(test_top5.avg))

    return test_losses.avg, val_losses.avg, test_top1.avg, val_top1.avg


if __name__ == '__main__':
    main()
