"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet, UCMercedDataset_finetune
from few_shot.core import NShotTaskSampler as TaskSampler, create_nshot_task_label, EvaluateFinetune, NShotTaskSampler
from few_shot.cnn_finetune import gradient_step
from few_shot.models import FewShotClassifier as FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH,CUDA

if __name__ == '__main__':
    setup_dirs()
    if CUDA:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='UCMerced')
    parser.add_argument('--n', default=1, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
    parser.add_argument('--inner-train-steps', default=1, type=int)
    parser.add_argument('--inner-val-steps', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--finetune-lr', default=0.1, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--order', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--epoch-len', default=1, type=int)
    parser.add_argument('--eval-batches', default=2, type=int)
    parser.add_argument('--eval-label', default='river,runway,sparseresidential,storagetanks,tenniscourt', type=str)

    args = parser.parse_args()

    if args.dataset == 'omniglot':
        dataset_class = OmniglotDataset
        fc_layer_size = 64
        num_input_channels = 1
    elif args.dataset == 'UCMerced':
        dataset_class = UCMercedDataset_finetune
        fc_layer_size = 16384
        num_input_channels = 3
    elif args.dataset == 'miniImageNet':
        dataset_class = MiniImageNet
        fc_layer_size = 1600
        num_input_channels = 3
    else:
        raise(ValueError('Unsupported dataset'))

    param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.batch_size}_' \
                f'eval={args.eval_label}'
    print(param_str)


    ###################
    # Create datasets #
    ###################
    background = dataset_class('background', **{'eval_label':args.eval_label})
    background_taskloader = DataLoader(
        background,
        batch_sampler=TaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                       num_tasks=args.batch_size),
        num_workers=8
    )
    evaluation = dataset_class('evaluation', **{'eval_label':args.eval_label})
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                       num_tasks=args.batch_size),
        num_workers=8
    )


    ############
    # Training #
    ############
    print(f'Training MAML on {args.dataset}...')
    from config import UCMerced_Label
    meta_model = FewShotClassifier(num_input_channels,len(UCMerced_Label), fc_layer_size).to(device, dtype=torch.double)

    meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss().to(device)


    def prepare_batch(n, k, q, meta_batch_size):
        def prepare_batch_(batch):
            x, y = batch
            # Reshape to `meta_batch_size` number of tasks. Each task contains
            # n*k support samples to train the fast model on and q*k query samples to
            # evaluate the fast model on and generate meta-gradients
            x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
            # Move to device
            x = x.double().to(device)
            # Create label
            y = y.reshape(meta_batch_size, n*k + q*k).to(device)
            return x, y

        return prepare_batch_


    callbacks = [
        EvaluateFinetune(
            eval_fn=gradient_step,
            num_tasks=args.eval_batches,
            n_shot=args.n,
            k_way=args.k,
            q_queries=args.q,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_batch(args.n, args.k, args.q, args.batch_size),
            # MAML kwargs
            inner_train_steps=args.inner_val_steps,
            inner_lr=args.finetune_lr,
            device=device,
            order=args.order,
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/finetune/{param_str}.pth',
            monitor=f'val_{args.n}-shot_{args.k}-way_acc'
        ),
        ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
        CSVLogger(PATH + f'/logs/finetune/{param_str}.csv'),
    ]


    fit(
        meta_model,
        meta_optimiser,
        loss_fn,
        epochs=args.epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_batch(args.n, args.k, args.q, args.batch_size),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=gradient_step,
        fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                             'train': True,
                             'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                             'inner_lr': args.finetune_lr},
    )
