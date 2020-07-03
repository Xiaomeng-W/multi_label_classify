import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=False, visual_size=2048):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func, self.bn, self.linear = self.__get_model()
        self.activation = nn.ReLU()
        self.W_fc = nn.Linear(in_features=visual_size*2, out_features=visual_size)

    def __init_weight(self):
        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def __get_model(self):
        model, out_features, func = None, None, None
        if self.model_name =='resnet50':
            resnet = models.resnet50(pretrained=self.pretrained)
            modules = list(resnet.children())[:-2]
            # print(len(modules))
            # for i in range(len(modules)):
            #     print(i, modules[i])
            model = nn.Sequential(*modules)
            # print(len(model))
            # print(model[0])
            out_features = resnet.fc.in_features
            func = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        linear = nn.Linear(in_features=out_features, out_features=out_features)
        bn = nn.BatchNorm1d(num_features=out_features, momentum=0.1)
        return model, out_features, func, bn, linear

    def similarity(self, x1, x2):
        lst = []
        for i in range(len(x1)):
            tmp = []
            for j in range(len(x1[i])):
                t1 = x1[i][j].flatten().detach().numpy()
                t2 = x2[i][j].flatten().detach().numpy()
                tmp.append(cosine_similarity([t1, t2]))
            lst.append(tmp)

    def get_params(self, params):
        for name, param in params:
            if param.requires_grad:
                print(name, param)

    def forward(self, image_cc, image_mlo, attention, h):
        # x1, x2 = image_cc, image_mlo
        # for i in range(len(self.model)):
        #     param = self.model[i].named_parameters()
        #     self.get_params(param)
        #     x1 = self.model[i](x1)
        #     x2 = self.model[i](x2)
        #     self.similarity(x1, x2)
        #
        # return x1, x2
        feature_cc = self.model(image_cc)
        feature_mlo = self.model(image_mlo)

        avg_feature_cc = self.avg_func(feature_cc).squeeze()
        avg_feature_mlo = self.avg_func(feature_mlo).squeeze()

        vcc_att = attention.forward(avg_feature_cc, h)
        vmlo_att = attention.forward(avg_feature_mlo, h)

        co_v = self.W_fc(torch.cat([vcc_att, vmlo_att], dim=1))
        return co_v

    def forward_cat(self, image_cc, image_mlo):
        feature_cc = self.model(image_cc)
        feature_mlo = self.model(image_mlo)
        
        feature = torch.cat([feature_cc, feature_mlo], dim=1)
        avg_feature = self.avg_func(feature).squeeze()

        return avg_feature

class MLC(nn.Module):
    def __init__(self, classes=10, fc_in_features=512, k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_feature):
        tags = self.softmax(self.classifier(avg_feature))
        return tags

class FeatureAttention(nn.Module):
    def __init__(self, visual_size=2048, hidden_size=512):
        super(FeatureAttention, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

    def forward(self, avg_features, h):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h.squeeze(1))
        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)
        return v_att
