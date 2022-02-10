import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from dataUtils import *
from model.model_hierarchical_twostream import *
from data import *

# from pycasper.name import Name
from BookKeeper import *
import wandb
os.environ['WANDB_API_KEY'] = 'bac3a003428df951a8e0b9e3878002a3227bbf0c'
os.environ['WANDB_DISABLE_CODE'] = 'true'  # Workaround cluster error


from argsUtils import argparseNloop
from slurmpy import Slurm
import numpy as np
from tqdm import tqdm
import copy
import pdb
import pickle


def train(args, exp_num):
    if args.load and os.path.isfile(args.load):
        load_pretrained_model = True
    else:
        load_pretrained_model = False
    args_subset = ['exp', 'cpk', 'model', 'time']

    # Loss hyper-params
    lmb = {k: getattr(args, k) for k in vars(args) if 'lmb_' in k}

    book = BookKeeper(args, args_subset, args_dict_update={'chunks': args.chunks,
                                                           'batch_size': args.batch_size,
                                                           'model': args.model,
                                                           's2v': args.s2v,
                                                           'cuda': args.cuda,
                                                           'save_dir': args.save_dir,
                                                           'early_stopping': args.early_stopping,
                                                           'debug': args.debug,
                                                           'stop_thresh': args.stop_thresh,
                                                           'desc': args.desc,
                                                           'curriculum': args.curriculum,
                                                           'lr': args.lr,
                                                           'lmb': lmb
                                                           },
                      # tensorboard=args.tb,
                      load_pretrained_model=load_pretrained_model)
    # load_pretrained_model makes sure that the model is loaded, old save files are not updated and _new_exp is called to assign new filename
    args = book.args

    # # Start Log
    book._start_log()
    wandb.init(project='Language2Motion', config=args.__dict__)

    # Training parameters
    path2data = args.path2data
    dataset = args.dataset
    lmksSubset = args.lmksSubset
    desc = args.desc
    split = (args.train_frac, args.dev_frac)
    idx_dependent = args.idx_dependent
    batch_size = args.batch_size
    time = args.time
    global chunks
    chunks = args.chunks
    offset = args.offset
    mask = args.mask
    feats_kind = args.feats_kind
    s2v = args.s2v
    f_new = args.f_new
    curriculum = args.curriculum

    if args.debug:
        shuffle = False
    else:
        shuffle = True

    # Load data iterables
    data_obj = None
    if args.dobj_path is None:
        data_obj = Data(path2data, dataset, lmksSubset, desc,
                    split, batch_size=batch_size,
                    time=time,
                    chunks=chunks,
                    offset=offset,
                    shuffle=shuffle,
                    mask=mask,
                    feats_kind=feats_kind,
                    s2v=s2v,
                    f_new=f_new)
        with open('dataProcessing/data_obj.pkl', 'wb') as outfile:
            pickle.dump(data_obj, outfile)
        print('Data Loaded and saved to disk.')
    else:
        with open(args.dobj_path, 'rb') as infile:
            data_obj = pickle.load(infile)

    train = data_obj.train
    dev = data_obj.dev
    test = data_obj.test

    # Create a model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_shape = data_obj.input_shape
    kwargs_keys = ['pose_size', 'trajectory_size']
    modelKwargs = {key: input_shape[key] for key in kwargs_keys}
    modelKwargs.update(args.modelKwargs)

    input_size = 4096  # the size of BERT

    discriminator = SymDiscriminator(vocab_size=args.vocab_size,
                                     embed_size=args.embed_size
                                     ).to(device).double()
    generator = SymPoseGenerator(chunks,
                                 input_size=input_size,
                                 vocab_size=args.vocab_size,
                                 Seq2SeqKwargs=modelKwargs,
                                 load=args.load).to(device).double()
    optim_gen = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    print('Model Created')
    print("Model's state_dict:")
    for param_tensor in generator.state_dict():
        print(param_tensor, "\t", generator.state_dict()[param_tensor].size())

    gan_loss_function = nn.BCELoss()

    # LR scheduler
    scheduler = lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    # Transforms
    columns = get_columns(feats_kind, data_obj)
    pre = None
    if feats_kind != 'symbol':
        pre = Transforms(args.transforms, columns, args.seed,
                     mask, feats_kind, dataset, f_new)

    def loop_train(generator, discriminator, data, epoch=0):
        running_loss = 0
        running_count = 0
        generator.train()
        discriminator.train()

        Tqdm = tqdm(data, desc='train' +
                    ' {:.10f}'.format(0), leave=False, ncols=50)
        for count, batch in enumerate(Tqdm):
            X, Y, s2v = batch['input'], batch['output'], batch['desc']
            pose, trajectory, start_trajectory = X
            pose_gt, trajectory_gt, start_trajectory_gt = Y

            if isinstance(s2v, torch.Tensor):
                s2v = s2v.to(device)

            x, y = None, None
            if feats_kind != 'symbol':
                x = torch.cat((trajectory, pose), dim=-1).to(device)
                y = torch.cat((trajectory_gt, pose_gt), dim=-1).to(device)

                # Transform before the model
                x = pre.transform(x)
                y = pre.transform(y)
                x = x[..., :-4]
                y = y[..., :-4]
            else:
                # TODO: Handle the subsampling better
                x = (pose/12).type(torch.cuda.LongTensor)
                y = (pose_gt/12).type(torch.cuda.LongTensor)

            # Discriminator training
            optim_dis.zero_grad()
            fake_samples_labels = torch.zeros(
                (batch_size, 1)).to(device='cuda').double()
            real_samples_labels = torch.ones(
                (batch_size, 1)).to(device='cuda').double()
            all_samples_labels = torch.cat(
                (real_samples_labels, fake_samples_labels))

            p_hat_l, _ = generator(x, y, s2v, lmb, train=True)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            all_samples = torch.cat((y.squeeze(), sym_hat_l))

            output_discriminator = discriminator(all_samples)
            loss_discriminator = 0.001 * \
                gan_loss_function(output_discriminator, all_samples_labels)
            # Tqdm.set_description(desc+'Discriminator Loss: {:.8f}'.format(loss_discriminator/running_count ))
            loss_discriminator.backward()
            optim_dis.step()

            # Generator training
            optim_gen.zero_grad()
            p_hat_l, internal_losses = generator(x, y, s2v, lmb, train=True)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            output_discriminator_generated = discriminator(sym_hat_l)
            loss_generator = 0.001 * \
                gan_loss_function(
                    output_discriminator_generated, real_samples_labels)

            loss_ = loss_generator.item()
            for loss_type in internal_losses:
                loss_generator += internal_losses[loss_type]
                loss_ += internal_losses[loss_type].item()

            running_count += np.prod(y.shape)
            running_loss += loss_

            # update tqdm
            Tqdm.set_description('Train Generator {:.8f} Discriminator {:.8f}'.format(
                running_loss/running_count, loss_discriminator))
            Tqdm.refresh()

            # These lines are required loss.backward and optimizer.step
            loss_generator.backward()
            optim_gen.step()

            x = x.detach()
            y = y.detach()
            loss_generator = loss_generator.detach()
            loss_discriminator = loss_discriminator.detach()
            p_hat_l = p_hat_l.detach()
            sym_hat_l = sym_hat_l.detach()

            # internal_losses = [i.detach() for i in internal_losses]
            if count >= 0 and args.debug:  # debugging by overfitting
                break

        return running_loss/running_count

    def loop_eval(generator, discriminator, data, epoch=0):
        running_loss = 0
        running_count = 0
        generator.eval()
        discriminator.eval()
        Tqdm = tqdm(data, desc='eval' +
                    ' {:.10f}'.format(0), leave=False, ncols=50)
        for count, batch in enumerate(Tqdm):
            X, Y, s2v = batch['input'], batch['output'], batch['desc']
            pose, trajectory, start_trajectory = X
            pose_gt, trajectory_gt, start_trajectory_gt = Y

            if isinstance(s2v, torch.Tensor):
                s2v = s2v.to(device)

            x, y = None, None
            if feats_kind != 'symbol':
                x = torch.cat((trajectory, pose), dim=-1).to(device)
                y = torch.cat((trajectory_gt, pose_gt), dim=-1).to(device)
                # Transform before the model
                x = pre.transform(x)
                y = pre.transform(y)
                x = x[..., :-4]
                y = y[..., :-4]
            else:
                # TODO: Handle the subsampling better
                x = (pose/12).type(torch.cuda.LongTensor)
                y = (pose_gt/12).type(torch.cuda.LongTensor)

            # Discriminator
            fake_samples_labels = torch.zeros(
                (batch_size, 1)).to(device='cuda').double()
            real_samples_labels = torch.ones(
                (batch_size, 1)).to(device='cuda').double()
            all_samples_labels = torch.cat(
                (real_samples_labels, fake_samples_labels))

            # yhat (fake samples) <= generator forward pass
            p_hat_l, _ = generator(x, y, s2v, lmb, train=False)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            all_samples = torch.cat((y.squeeze(), sym_hat_l))

            output_discriminator = discriminator(all_samples)
            loss_discriminator = 0.001 * \
                gan_loss_function(output_discriminator, all_samples_labels)

            # Generator
            p_hat_l, internal_losses = generator(x, y, s2v, lmb, train=False)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            output_discriminator_generated = discriminator(sym_hat_l)
            loss_generator = 0.001 * \
                gan_loss_function(
                    output_discriminator_generated, real_samples_labels)

            loss_ = loss_generator.item()
            for loss_type in internal_losses:
                loss_generator += internal_losses[loss_type]
                loss_ += internal_losses[loss_type].item()

            running_count += np.prod(y.shape)
            running_loss += loss_
            # update tqdm
            Tqdm.set_description('Validation Generator {:.8f} : Discriminator {:.8f}'.format(
                running_loss/running_count, loss_discriminator))
            Tqdm.refresh()

            x = x.detach()
            y = y.detach()
            loss_generator = loss_generator.detach()
            loss_discriminator = loss_discriminator.detach()
            p_hat_l = p_hat_l.detach()
            sym_hat_l = sym_hat_l.detach()

            # internal_losses = [i.detach() for i in internal_losses]
            if count >= 0 and args.debug:  # debugging by overfitting
                break

        return running_loss/running_count

    num_epochs = args.num_epochs
    time_list = []
    time_list_idx = 0
    if curriculum:
        for power in range(1, int(np.log2(time-1)) + 1):
            time_list.append(2**power)
        data.update_dataloaders(time_list[0])
    time_list.append(time)
    tqdm.write('Training up to time: {}'.format(time_list[time_list_idx]))

    # Training Loop
    for epoch in tqdm(range(num_epochs), ncols=50):
        train_loss = loop_train(generator, discriminator, train, epoch=epoch)
        dev_loss = loop_eval(generator, discriminator, dev, epoch=epoch)
        test_loss = loop_eval(generator, discriminator, test, epoch=epoch)
        scheduler.step()  # Change the Learning Rate

        # save results
        book.update_res({'epoch': epoch, 'train': train_loss,
                         'dev': dev_loss, 'test': test_loss})
        book._save_res()

        # print results
        book.print_res(epoch, key_order=[
            'train', 'dev', 'test'], exp=exp_num, lr=scheduler.get_last_lr())
        if book.stop_training(generator, epoch):
            # if early_stopping criterion is met,
            # start training with more time steps
            time_list_idx += 1
            book.stop_count = 0  # reset the threshold counter
            book.best_dev_score = np.inf
            generator.load_state_dict(copy.deepcopy(book.best_model))
            if len(time_list) > time_list_idx:
                time_ = time_list[time_list_idx]
                data_obj.update_dataloaders(time_)
                tqdm.write('Training up to time: {}'.format(time_))


if __name__ == '__main__':
    argparseNloop(train)
