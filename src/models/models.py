"""
Written by Lincolnzjx & Jiabin
"""
#pylint: disable=W0223

import os
import sys
import itertools
import shutil
#from collections import OrderedDict
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

from models.DatasetFolder import DatasetFolder
from models import custom_loss
from models import grad


class ConvNetTrain(nn.Module):
    """Main class to train and generate result."""

    def __init__(self, net, args, logger):
        super(ConvNetTrain, self).__init__()
        self.args = args
        self.logger = logger

        if torch.cuda.is_available():
            cuda = self.args.cuda.split(',')
            device_ids = list(range(len(cuda)))
            self.logger.info('=> Using GPU device_ids {}'.format(*device_ids))
            self.net = net.cuda()
            #TODO make model could be deployed on multi gpu
            #self.net = nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            self.logger.error('=> No avaiable GPU')
            sys.exit(-1)
        #self.best_pred = 0.0

        # Loss
        #self.logger.info('=> Criterion: weight ce_loss')
        #weights = [0.0395, 0.0066, 0.0856, 0.1348, 0.0400, 0.3823, 0.3113]
        #self.class_weights = torch.FloatTensor(weights).cuda()
        #self.criterion_ce = nn.CrossEntropyLoss(weight=self.class_weights).cuda()

        self.logger.info('=> Criterion: ce_loss')
        self.criterion_ce = nn.CrossEntropyLoss().cuda()

        self.criterion_grad = custom_loss.GradLoss(self.args.inner_threshold,  \
                self.logger).cuda()

        # Optmizier
        #self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, amsgrad=True)


        # optimizer
        self.optimizer = optim.SGD([{'params': self.net.features.parameters(), \
                                            'lr': self.args.lr},
                                    {'params': self.net.classifier.parameters(), \
                                            'lr': self.args.lr*10, 'weight_decay':0.0005}])
                                            #'lr': self.args.lr, 'weight_decay':0.0005}])
                                   #lr=self.args.lr,
                                   #momentum=0.9,
                                   #nesterov=True,
                                   #weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
                                                              milestones=[50, 100, 200, 300], \
                                                              gamma=0.1)
        self.logger.info('using optim {} with init_lr: {}'.format('SGD', self.args.lr))

        self.target_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        # load model
        #self.load_model()


    def load_model(self, stage, stage_one, stage_two):
        """load model
        Args:
            stage
            stage_one
            stage_two
        Return:
        """
        model_path = '../train_dir'
        if stage == 1 and stage_one == 'pretrained':
            self.logger.info('=> Stage {}, Load model from pretrained'.format(stage))
            return
        elif stage == 1:
            self.logger.info('=> Stage {}, Load model from {}'.format(stage, stage_one))
            model_path = os.path.join(model_path, stage_one)
        elif stage == 2:
            self.logger.info('=> Stage {}, Load model from {}'.format(stage, stage_two))
            model_path = os.path.join(model_path, stage_two)

        checkpoint = torch.load(model_path)
        model_parameters = checkpoint['state_dict']
        self.net.load_state_dict(model_parameters)
        self.logger.debug('Load model succeed from {}'.format(model_path))
        #dd=[(x.split('.', maxsplit=1)[-1], y) for x, y in d.items()]
        #ddd=OrderedDict(sorted(dd, key=lambda t: t[0]))


    def train(self, epoch, trainloader, ntrain, stage):
        """Train Network
        Args:
            epoch
            trainloader
            ntrain
            stage
        Return:
            train_ce_losses
            train_cam_losse
            correct
            predicted
        """
        alpha = torch.tensor(self.args.alpha).cuda()

        self.logger.info('\nEpoch: %d' %epoch)
        self.net.train()

        train_cam_losses = 0
        train_ce_losses = 0
        y_true = []
        y_pred = []

        self.scheduler.step()
        for batch_idx, (inputs, targets, has_masks, inner_batches, outer_batches)\
            in enumerate(trainloader):

            inputs, targets = inputs.cuda(), targets.cuda()

            self.optimizer.zero_grad()

            features, outputs = self.net.extractor(inputs)
            ce_losses = self.criterion_ce(outputs, targets)
            _, pred = torch.max(outputs.data, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            if stage == 2:
                targets_onehot = self.one_hot(targets, len(self.target_names))
                targets_value = targets_onehot * outputs
                sum_output = torch.sum(targets_value)

                sum_output.backward(retain_graph=True)
                grad_vals = self.net.get_gradients()[-1]

                # LOSS: loss + loss+grad
                index = 0
                cam_losses = torch.tensor(0.0, dtype=torch.float32).cuda()
                for has_mask, feature, grad_val, inner, outer in  \
                        zip(has_masks, features[0], grad_vals, inner_batches, outer_batches):
                    if has_mask:
                        cam = grad.grad(feature, grad_val).cuda()
                        inner, outer = inner.cuda(), outer.cuda()
                        cam_loss = self.criterion_grad(cam, inner, outer).cuda()
                        cam_losses = cam_losses + cam_loss
                        #self.logger.debug('loss: {}'.format(cam_loss))
                    index = index + 1

                total_losses = alpha * cam_losses + (1-alpha) * ce_losses
                #self.optimizer.zero_grad()
                total_losses.backward()
                self.optimizer.step()

                train_cam_losses += alpha * cam_losses.item()
                train_ce_losses += (1-alpha.item()) *  ce_losses.item()
                print('{} ceLoss:{} camLoss:{}'.format(batch_idx+1, ce_losses.item(), cam_losses.item()))
            else:
                train_cam_losses = 0.0
                train_ce_losses += ce_losses.item()
                total_losses = ce_losses
                #self.optimizer.zero_grad()
                ce_losses.backward()
                self.optimizer.step()

            if (batch_idx+1) % 10 == 0:
                #self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' \
                #       % (epoch, self.args.n_epochs, batch_idx+1,  \
                #           ntrain // self.args.batch_size, total_losses.item()))
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] camLoss: %.4f' \
                       % (epoch, self.args.n_epochs, batch_idx+1,  \
                           ntrain // self.args.batch_size, train_cam_losses))
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] ceLoss: %.4f' \
                       % (epoch, self.args.n_epochs, batch_idx+1,  \
                           ntrain // self.args.batch_size, train_ce_losses))


        #acc, mca, class_precision = self.getMCA(correct, predicted)
        return train_ce_losses, train_cam_losses, y_true, y_pred


    #def only_test(self):
    #    """only test but not train"""
    #    if os.path.exists(self.args.train_dir) is False:
    #        os.mkdir(self.args.train_dir)
    #    accTe, mcaTe, class_precision_test, correct, predicted = self.test(0)


    def test(self, epoch, testloader, ntest):
        """test model one time on validation dataset
        Args:
            epoch:
            testloader:
            ntest
        Return:
            correct
            predicted
        """
        self.net.eval()
        y_true = []
        y_pred = []
        y_pred_scores = []
        test_losses = 0.0

        self.logger.info('=>Epoch: {} Testing '.format(epoch))

        # check grad set zero
        for p in self.net.parameters():
            if p.grad is not None:
                self.logger.debug("{} {}".format(p.name, p.grad))

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = self.net(inputs)
                outputs_prob = torch.nn.functional.softmax(outputs, dim=1).data.cpu().numpy()
                test_loss = self.criterion_ce(outputs, targets)
                test_losses += test_loss.item()


            _, pred = torch.max(outputs.data, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            for index, i in enumerate(targets.cpu().numpy()):
                y_pred_scores.append(outputs_prob[index][i])

            #y_pred_score.extend(pred_score.cpu().numpy())
            self.logger.debug('y_pred len: {}, {}'.format(len(y_pred_scores), y_pred_scores))


            if (batch_idx+1) % 10 == 0:
                self.logger.info('Completed: [%d/%d], Loss: %.4f' %(batch_idx+1, ntest//self.args.batch_size, test_loss.item()))

        return test_losses, y_true, y_pred, y_pred_scores


    def get_average_class_precision(self, y_true, y_pred, target_names):
        """get average class precision
        Args:
            y_ture
            y_pred
            target_names
        Returns:
            avarage_class_preicision
            accuracy
            class_precision
        """
        avarage_class_preicision = precision_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        class_precision = []
        #for x in class_report.key
        #class_precision = [class_report['precision'] for class_name, class_report \
        #        in zip(target_names, report)]
        #class_precision = report
        for class_name, class_report in zip(target_names, report):
            self.logger.info('=> class_name, class_report {}: {}'.format(class_name, class_report))

        #self.logger.debug(report)
        self.logger.info(avarage_class_preicision)
        #sys.exit(-1)

        self.logger.info('=> classification_report \n{}'.format(report))
        return avarage_class_preicision, accuracy, class_precision

        #for class_index in range(classes_length):
        #    count = 0.0
        #    tot = 0.0
        #    for i, x in enumerate(correct):
        #        if x == class_index:
        #            tot += 1
        #            if x == predicted[i]:
        #                count += 1

        #    acc_t = count/tot*100.0
        #    mca = mca + acc_t
        #    class_precision.append(acc_t)

        #mca = mca/len(self.class_weights)

        #acc = 0
        #for i, x in enumerate(correct):
        #    if x == predicted[i]:
        #        acc = acc+1

        #acc = acc/len(predicted)*100

        #return acc, mca, class_precision


    #def get_mean_std(self):
    #    """get mean and std
    #    """
    #    filename = '../data/split_data/mean_std_shuffle.csv'
    #    #csvfile = pd.read_csv(filename, index_col=0).values[int(self.args.iterNo)-1]
    #    csvfile = pd.read_csv(filename).values[int(self.args.iterNo)-1]
    #    print(csvfile)
    #    return csvfile[0:3], csvfile[3:]

    def get_loaders(self, stage):
        """get data loader
        """
        #resize_img = 300
        img_size = 224
        #mean, std = self.get_mean_std()
        # mean std from resnet50 on imagenet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalized = transforms.Normalize(mean=mean, std=std)
        #if stage == 1:
        #    transform_train = transforms.Compose([
        #        transforms.Resize((img_size, img_size)),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.RandomVerticalFlip(),
        #        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        #        transforms.RandomRotation([-180, 180]),
        #        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
        #        transforms.ToTensor(),
        #        normalized
        #    ])
        #elif stage == 2:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalized
        ])
        #else:
        #    self.logger.error('=>Error, Stage has to be set')
        #    sys.exit(-1)

        self.logger.info('=> Preparing data..')

        trainset = DatasetFolder(train=True, transform=transform_train, \
                iter_no=int(self.args.iterNo), logger=self.logger)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.args.batch_size,
                                                  num_workers=20,
                                                  shuffle=True)

        # test
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalized
        ])

        testset = DatasetFolder(train=False, transform=transform_test, \
                iter_no=int(self.args.iterNo), logger=self.logger)

        testloader = torch.utils.data.DataLoader(testset, \
                batch_size=self.args.batch_size, \
                num_workers=5, shuffle=False)

        return trainloader, testloader, len(trainset), len(testset)


    def iterate_convnetwork(self):
        """Train CNN here
        """
        best_test_acp = 0.0
        # load data
        trainloader, testloader, ntrain, ntest = self.get_loaders(self.args.stage)

        # load model
        self.load_model(self.args.stage, self.args.stage_one, self.args.stage_two)
        #best_test_mca = 0.0 if self.best_pred == None else self.best_pred

        # summary
        output_writer_path = os.path.join('../tensorboard_logfile', self.args.logfile)
        writer = SummaryWriter(output_writer_path)
        if os.path.exists('../train_dir') is False:
            os.mkdir('../train_dir')

        for epoch in range(self.args.start_epoch, self.args.n_epochs+1):
            self.logger.info('{}'.format(self.args.bash_name))
            for index, params in enumerate(self.optimizer.state_dict()['param_groups']):
                writer.add_scalar('train/lr_' + str(index+1), params['lr'], epoch)
            #for name, param in self.net.named_parameters():
            #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            # TRAIN
            train_ce_losses, train_cam_losses, y_true, y_pred = self.train(epoch, \
                    trainloader, ntrain, self.args.stage)

            train_acp, train_acc, _ = self.get_average_class_precision( \
                    y_true, y_pred, self.target_names)
            fig, title = self.plot_confusion_matrix(y_true, y_pred, self.target_names)
            writer.add_figure('Train_'+title, fig, epoch)

            writer.add_scalar('train/ce_loss', train_ce_losses, epoch)
            writer.add_scalar('train/cam_loss', train_cam_losses, epoch)
            writer.add_scalar('train/total_loss', train_cam_losses+train_ce_losses, epoch)
            writer.add_scalar('train/acp', train_acp, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)

            # TEST
            test_losses, y_true, y_pred, y_pred_scores = self.test(epoch, testloader, ntest)
            self.draw_auc(y_pred_scores, y_true, writer, epoch)
            test_acp, test_acc, _ = self.get_average_class_precision(\
                    y_true, y_pred, self.target_names)

            fig, title = self.plot_confusion_matrix(y_true, y_pred, self.target_names)
            writer.add_scalar('valid/ce_loss', test_losses, epoch)
            writer.add_figure('valid_' + title, fig, epoch)
            writer.add_scalar('valid/acc', test_acc, epoch)
            writer.add_scalar('valid/acp', test_acp, epoch)

            if epoch % 20 == 0:
                if test_acp > best_test_acp:
                    best_test_acp = test_acp
                    is_best = True
                else:
                    is_best = False

                train_dir = os.path.join('../train_dir', self.args.train_dir)
                if os.path.exists(train_dir) is False:
                    os.mkdir(train_dir)
                path = os.path.join(train_dir, str(epoch))
                self.save_checkpoint({
                    'epoch': epoch+1,
                    'arch': self.args,
                    'state_dict':self.net.state_dict(),
                    'best_pred': best_test_acp,
                    'optimizer': self.optimizer.state_dict()
                    }, is_best, path)


    def plot_confusion_matrix(self, y_true, y_pred, class_names, normalized=True):
        """plot confusion matrix"""
        self.logger.debug("=> plot_confusion_matrix")
        classes = class_names
        cf_matrix = confusion_matrix(y_true, y_pred)
        title = 'confusion_matrix_{}'
        if normalized is True:
            cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
            title = title.format('Normalize')
        else:
            title.format('Not Normalize')

        np.set_printoptions(precision=2)

        fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cf_matrix, cmap='Oranges')


        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=4, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
            ax.text(j, i, format(cf_matrix[i, j], 'f') if cf_matrix[i, j] != 0 else '.', \
                    horizontalalignment="center", fontsize=6, \
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)

        return fig, title

    def save_checkpoint(self, state, is_best, filename):
        """save checkpoint
        """
        self.logger.debug('=> save_checkpoint')
        torch.save(state, filename)
        if is_best:
            dirname = os.path.dirname(filename)
            shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))

    def one_hot(self, y_true, digits):
        """one hot"""
        self.logger.debug("=> one_hot")
        batch_size = y_true.size()[0]
        y_true = y_true.view(batch_size, 1)
        y_true_onehot = torch.FloatTensor(batch_size, digits).cuda()
        y_true_onehot.zero_()
        y_true_onehot.scatter_(1, y_true, 1)
        return y_true_onehot

    def draw_auc(self, y_pred, y_true, writer, epoch):
        """draw_auc
        Args:
            y_pred: actually y_pred_score
        """

        for index, class_name in enumerate(self.target_names):
            self.logger.debug('Draw {} {} auc'.format(index, class_name))
            index_y_pred = y_pred.copy()
            index_y_true = y_true.copy()
            for iindex, the_y_true in enumerate(index_y_true):
                if the_y_true != index:
                    index_y_true[iindex] = -1
            auc, _, _, _ = self.get_auc(index_y_pred, index_y_true, index)
            writer.add_scalar('valid/auc_class_{}'.format(index), auc, epoch)


    def get_auc(self, y_pred, y_true, pos_label):
        """get auc"""
        self.logger.debug('=> Get_AUC')
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=pos_label)
        auc = sklearn.metrics.auc(fpr, tpr)
        return auc, fpr, tpr, thresholds
