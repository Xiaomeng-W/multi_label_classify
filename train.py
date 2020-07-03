import argparse
import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from utils.dataset import *
from utils.models import *
from utils.logger import *

class TrainBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_val_tag_loss = 1000000

        self.min_train_loss = 10000000000
        self.min_train_tag_loss = 1000000

        self.params = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.model_state_dict = self._load_model_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_cc, self.args.train_file_mlo, self.train_transform)
        self.val_data_loader = self._init_data_loader(self.args.val_file_cc, self.args.val_file_mlo, self.val_transform)

        self.extractor = self._init_visual_extractor()
        self.attention = self._init_attention()
        self.mlc = self._init_mlc()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_schedule()
        self.logger = self._init_logger()
        self.writer.write("{}\n".format(self.args))

    def train(self):
        for epoch_id in range(self.start_epoch, self.args.epochs):
            train_tag_loss = self._epoch_train()
            val_tag_loss = self._epoch_val()
            print(epoch_id)
            print(train_tag_loss)
            print(val_tag_loss)

            if self.args.mode == 'train':
                self.scheduler.step(train_tag_loss)
            else:
                self.scheduler.step(val_tag_loss)
            self.writer.write('[{} - Epoch {}] train_tag_loss:{} - val_tag_loss:{} - lr:{}\n'.format(self._get_now(),
                                                                                                     epoch_id,
                                                                                                     train_tag_loss,
                                                                                                     val_tag_loss,
                                                                                                     self.optimizer.param_groups[0]['lr']))
            self._save_model(epoch_id, val_tag_loss, train_tag_loss)
            self._log(train_tags_loss=train_tag_loss, val_tags_loss=val_tag_loss, lr=self.optimizer.param_groups[0]['lr'], epoch=epoch_id)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, self._get_now().replace(':','-'))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'log.txt'), 'w')
        return writer

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _load_model_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            self.writer.write('[Load model-{} succeed!]\n'.format(self.args.load_model_path))
            self.writer.write('Load from epoch {}\n'.format(model_state['epoch']))
            return model_state
        except Exception as err:
            self.writer.write('[Load model falied] {}\n'.format(err))
            return None

    def _init_data_loader(self, file_list_cc, file_list_mlo, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 file_list_cc=file_list_cc,
                                 file_list_mlo=file_list_mlo,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=True)
        return data_loader

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained,
                                       visual_size = 2048)
        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write('[Load feature extractor succeed!]\n')
        except Exception as err:
            self.writer.write('[Load feature extractor failed] {}\n'.format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_attention(self):
        model = FeatureAttention(visual_size=self.extractor.out_features, hidden_size=self.args.hidden_size)
        try:
            model_state = torch.load(self.args.load_attention_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write('[Load Attention model succeed!]\n')
        except Exception as err:
            self.writer.write('[Load Attention model failed {}]\n'.format(err))

        if not self.args.att_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_mlc(self):
        model = MLC(classes=self.args.classes, fc_in_features=self.extractor.out_features, k=self.args.k)
        try:
            model_state = torch.load(self.args.load_mlc_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write('[Load MLC succeed!]\n')
        except Exception as err:
            self.writer.write('[Load MLC failed {}]\n'.format(err))

        if not self.args.mlc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _init_schedule(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_logeer(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _save_model(self, epoch_id, val_tag_loss, train_tag_loss):
        def save_whole_model(_filename):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'mlc': self.mlc.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        if val_tag_loss < self.min_val_tag_loss:
            file_name = "val_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_val_tag_loss = val_tag_loss

        if train_tag_loss < self.min_train_tag_loss:
            file_name = "train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_train_tag_loss = train_tag_loss

    def _log(self,
             train_tags_loss,
             val_tags_loss,
             lr,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'val tags loss': val_tags_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

class Trainer(TrainBase):
    def __init__(self, args):
        TrainBase.__init__(self, args)
        self.args = args

    def _epoch_train(self):
        tag_loss = 0
        self.extractor.train()
        self.mlc.train()
        self.attention.train()

        for i, (images_cc, images_mlo, _, _, label_cc, label_mlo) in enumerate(self.train_data_loader):
            batch_tag_loss = 0
            images_cc = self._to_var(images_cc)
            images_mlo = self._to_var(images_mlo)
            label_cc = self._to_var(label_cc, requires_grad=False)

            h = self._to_var(torch.zeros(images_cc.shape[0], 1, self.args.hidden_size))
            co_v = self.extractor.forward(images_cc, images_mlo, self.attention, h)
            tags = self.mlc.forward(co_v)
            batch_tag_loss = self.mse_criterion(tags, label_cc).sum()

            self.optimizer.zero_grad()
            batch_tag_loss.backward()

            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm(self.extractor.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.mlc.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.attention.parameters(), self.args.clip)
            self.optimizer.step()

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
        return tag_loss

    def _epoch_val(self):
        tag_loss = 0
        self.extractor.eval()
        self.mlc.eval()
        self.attention.eval()

        for i, (images_cc, images_mlo, _, _, label_cc, label_mlo) in enumerate(self.val_data_loader):
            images_cc = self._to_var(images_cc, requires_grad=False)
            images_mlo = self._to_var(images_mlo, requires_grad=False)
            label_cc = self._to_var(label_cc, requires_grad=False)

            h = self._to_var(torch.zeros(images_cc.shape[0], 1, self.args.hidden_size))
            co_v = self.extractor.forward(images_cc, images_mlo, self.attention, h)
            tags = self.mlc.forward(co_v)
            batch_tag_loss = self.mse_criterion(tags, label_cc).sum()

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
        return tag_loss

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    #Data Argument
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--train_file_cc', type=str, default='./data/train_data_cc.txt',
                        help='the train_cc one hot array')
    parser.add_argument('--train_file_mlo', type=str, default='./data/train_data_mlo.txt',
                        help='the train_mlo one hot array')
    parser.add_argument('--val_file_cc', type=str, default='./data/val_data_cc.txt',
                        help='the test_cc one hot array')
    parser.add_argument('--val_file_mlo', type=str, default='./data/val_data_mlo.txt',
                        help='the test_mlo one hot array')

    # Transform Argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size of randomly cropping images')

    # Save/Load model Argument
    parser.add_argument('--model_path', type=str, default='./models',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='the path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='breast_model',
                        help='the name of saved model')

    # Model Argument
    parser.add_argument('--momentum', type=int, default=0.1)

    # Feature Extractor
    parser.add_argument('--visual_model_name', type=str, default='resnet50',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str, default='.')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='whether train visual extractor or not')

    # Attention
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_attention_model_path', type=str, default='.')
    parser.add_argument('--att_trained', action='store_true', default=True)

    # MLC
    parser.add_argument('--classes', type=int, default=9)
    parser.add_argument('--k', type=int, default=9)
    parser.add_argument('--load_mlc_model_path', type=str, default='.')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # Training Argument
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: 0.35)')

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    trainer = Trainer(args)
    trainer.train()