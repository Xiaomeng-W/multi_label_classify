import pickle
import argparse
from tqdm import tqdm
import cv2
import json

import torchvision.transforms as transforms
from torch.autograd import Variable

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.build_tag import *


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args
        self.params = None

        self.tagger = self.__init_tagger()
        self.transform = self.__init_transform()
        self.data_loader = self.__init_data_loader(self.args.test_file_list_cc, self.args.test_file_list_mlo)
        self.model_state_dict = self.__load_mode_state_dict()

        self.extractor = self.__init_visual_extractor()
        self.attention = self._init_attention()
        self.mlc = self.__init_mlc()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def test(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.eval()
        self.attention.eval()
        self.mlc.eval()
        for i, (images_cc, images_mlo, _, _, label_cc, label_mlo) in enumerate(self.data_loader):
            batch_tag_loss = 0
            images_cc = self._to_var(images_cc)
            images_mlo = self._to_var(images_mlo)
            label_cc = self._to_var(label_cc, requires_grad=False)

            co_v = self.extractor.forward_cat(images_cc, images_mlo)
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

    def generate(self):
        self.extractor.eval()
        self.attention.train()
        self.mlc.eval()

        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}

        for images_cc, images_mlo, image_cc_id, image_mlo_id, label_cc, label_mlo in progress_bar:
            images_cc = self.__to_var(images_cc, requires_grad=False)
            images_mlo = self.__to_var(images_mlo, requires_grad=False)
            co_v = self.extractor.forward_cat(images_cc, images_mlo)
            tags = self.mlc.forward(co_v)

            pred_sentences = {}
            real_sentences = {}
            for i in image_cc_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            for id, pred_tag, real_tag in zip(image_cc_id, tags, label_cc):
                real = self.tagger.inv_tags2array(real_tag)
                k = len(real)
                results[id] = {
                    'Real Tags': real,
                    'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, k)[1].cpu().detach().numpy()),
                }

        self.__save_json(results)

    def _generate_cam(self, images_id, visual_features, alpha_v, sentence_id):
        alpha_v *= 100
        cam = torch.mul(visual_features, alpha_v.view(alpha_v.shape[0], alpha_v.shape[1], 1, 1)).sum(1)
        cam.squeeze_()
        cam = cam.cpu().data.numpy()
        for i in range(cam.shape[0]):
            image_id = images_id[i]
            cam_dir = self.__init_cam_path(images_id[i])

            org_img = cv2.imread(os.path.join(self.args.image_dir, image_id), 1)
            org_img = cv2.resize(org_img, (self.args.cam_size, self.args.cam_size))

            heatmap = cam[i]
            heatmap = heatmap / np.max(heatmap)
            heatmap = cv2.resize(heatmap, (self.args.cam_size, self.args.cam_size))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            img = heatmap * 0.5 + org_img
            cv2.imwrite(os.path.join(cam_dir, '{}.png'.format(sentence_id)), img)

    def __init_cam_path(self, image_file):
        generate_dir = os.path.join(self.args.model_dir, self.args.generate_dir)
        if not os.path.exists(generate_dir):
            os.makedirs(generate_dir)

        image_dir = os.path.join(generate_dir, image_file)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __save_json(self, result):
        result_path = os.path.join(self.args.model_dir, self.args.result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(os.path.join(self.args.model_dir, self.args.load_model_path), map_location='cpu')
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_tagger(self):
        return Tag()


    def __init_data_loader(self, file_list_cc, file_list_mlo):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 file_list_cc=file_list_cc,
                                 file_list_mlo=file_list_mlo,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=False)
        return data_loader

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)

        if self.model_state_dict is not None:
            print("Visual Extractor Loaded!")
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def _init_attention(self):
        model = FeatureAttention(visual_size=self.extractor.out_features, hidden_size=self.args.hidden_size)
        try:
            model_state = torch.load(self.args.load_attention_model_path)
            model.load_state_dict(model_state['model'])
            print('[Load Attention model succeed!]\n')
        except Exception as err:
            print('[Load Attention model failed {}]\n'.format(err))

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

    def __init_mlc(self):
        model = MLC(classes=self.args.classes,
                    fc_in_features=self.extractor.out_features*2,
                    k=self.args.k)

        if self.model_state_dict is not None:
            print("MLC Loaded!")
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()

        return model

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    # Path Argument
    parser.add_argument('--model_dir', type=str, default='./models/breast_model/20200624-03-16')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--test_file_list_cc', type=str, default='./data/test_data_cc.txt',
                        help='the test_cc array')
    parser.add_argument('--test_file_list_mlo', type=str, default='./data/train_data_mlo.txt',
                        help='the test_mlo array')
    parser.add_argument('--load_model_path', type=str, default='train_best_loss.pth.tar',
                        help='The path of loaded model')

    # transforms argument
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resizing images')

    # CAM
    parser.add_argument('--cam_size', type=int, default=224)
    parser.add_argument('--generate_dir', type=str, default='cam')

    # Saved result
    parser.add_argument('--result_path', type=str, default='results',
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='debug',
                        help='the name of results')

    """
    Model argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet50',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')

    # MLC
    parser.add_argument('--classes', type=int, default=9)
    parser.add_argument('--k', type=int, default=9)

    # Co-Attention
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--att_trained', action='store_true', default=True)

    """
    Generating Argument
    """
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=8)

    # Loss function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print(args)

    sampler = CaptionSampler(args)

    sampler.generate()
