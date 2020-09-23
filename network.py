import torch
import torch.nn as nn
import itertools
from util import *
from torchvision import transforms, models
import numpy as np
from torch.nn.utils import weight_norm
args = get_args()

class G_u(nn.Module): #
    def __init__(self, args):
        super(G_u, self).__init__()

        self._relu = nn.ReLU()
        #self._dropout = nn.Dropout(args.dropout)
        self._ws1 = nn.Linear(args.video_feature_dim, args.Vu_middle_feature_dim, bias=False)
        self._ws2 = nn.Linear(args.Vu_middle_feature_dim, args.image_feature_dim, bias=False)

        self._init_weights()
    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)
        self._ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, video_input):
        # video input is [batch_size, clip_num, feature_dim]    1024 -> 2048 -> RELU -> 2048 - > RELU
        video_size = video_input.size() # [bsz, num, dim]
        video_compressed_embeddings = video_input.view(-1, video_size[2]) # [bsz * num, dim]
        v_u = self._relu(self._ws1(video_compressed_embeddings)) #[bsz * num, mid_dim]
        fake_image = self._relu(self._ws2(v_u)).view(video_size[0], video_size[1], -1)   #[bsz, num , res_dim]

        return fake_image


class G_t(nn.Module):
    def __init__(self, args):
        super(G_t, self).__init__()

        self._relu = nn.ReLU()
        # self._dropout = nn.Dropout(args.dropout)
        self._ws1 = nn.Linear(args.image_feature_dim, args.Vt_middle_feature_dim, bias=False)
        self._ws2 = nn.Linear(args.Vt_middle_feature_dim, args.video_feature_dim, bias=False)

        self._init_weights()
    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)
        self._ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, image_input):
        # image input is [batch_size, clip_num, feature_dim]
        image_size = image_input.size() # [bsz, num, dim]
        image_compressed_embeddings = image_input.view(-1, image_size[2]) # [bsz * num, dim]
        v_t = self._relu(self._ws1(image_compressed_embeddings)) #[bsz * num, mid_dim]
        fake_video = self._relu(self._ws2(v_t)).view(image_size[0], image_size[1], -1)   #[bsz, num, res_dim]

        return fake_video


class D_V(nn.Module):
    def __init__(self, args):
        super(D_V, self).__init__()

        self._relu = nn.ReLU()
        #self._dropout = nn.Dropout(args.dropout)

        self._ws1 = nn.Linear(args.video_feature_dim, args.DV_middle_feature_dim, bias=False)
        self._ws2 = nn.Linear(args.DV_middle_feature_dim, 1, bias=False)

        self._init_weights()
    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)
        self._ws2.weight.data.uniform_(-init_range, init_range)


    def forward(self, video_input):
        # video input is [batch_size, clip_num, feature_dim]    1024 -> 256 -> RELU -> 1
        video_size = video_input.size() # [bsz, num, dim]
        video_compressed_embeddings = video_input.view(-1, video_size[2]) # [bsz * num, dim]
        dv_middle = self._relu(self._ws1(video_compressed_embeddings)) #[bsz * num, mid_dim]
        video_logit = self._ws2(dv_middle).view(video_size[0], video_size[1], -1) #[bsz, num , res_dim]

        return video_logit

class f(nn.Module): #
    def __init__(self, args):
        super(f, self).__init__()

        self._relu = nn.ReLU()
        #self._softmax = nn.Softmax(dim= -1)
        #self._dropout = nn.dropout(args.dropout)

        #self._ws1 = nn.Linear(args.image_feature_dim + args.video_feature_dim, args.f_middle1_feature_dim, bias=False)
        self._ws1 = nn.Linear(args.video_feature_dim + args.image_feature_dim , args.f_middle1_feature_dim, bias=False)
        self._ws2 = nn.Linear(args.f_middle1_feature_dim, args.f_middle2_feature_dim, bias=False)
        self._ws3 = nn.Linear(args.f_middle2_feature_dim, args.f_class_dim, bias=False)
        #self._ws2 = nn.Linear(args.f_middle1_feature_dim, args.f_class_dim, bias=False)

        self._init_weights()
    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)
        self._ws2.weight.data.uniform_(-init_range, init_range)
        self._ws3.weight.data.uniform_(-init_range, init_range)

    def forward(self, input):
        # input is [batch_size, clip_num, feature_dim]    (2048+1024) -> 1024 -> RELU -> 512 - > RELU -> 20
        input_size = input.size() # [bsz, num, dim]
        image_compressed_embeddings = input.view(-1, input_size[2]) # [bsz * num, dim]
        f_middle1 = self._relu(self._ws1(image_compressed_embeddings)) #[bsz * num, mid_dim1]
        f_middle2 = self._relu(self._ws2(f_middle1))   #[bsz*num, mid_dim2]
        f_class = self._ws3(f_middle2).view(input_size[0], input_size[1], -1)
        #f_class = self._ws2(f_middle1).view(input_size[0], input_size[1], -1)
        return f_class


class Att(nn.Module): # 1024 - > 1 new 2048-> 1
    def __init__(self, args):
        super(Att, self).__init__()

        #self._softmax = nn.Softmax(dim=0)
        self._sigmoid = nn.Sigmoid()
        # self._dropout = nn.Dropout(args.dropout)
        self._ws1 = nn.Linear(args.video_feature_dim, 1, bias=False)

        self._init_weights()
    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)

    def forward(self, video_input):
        # input is [batch_size, clip_num, feature_dim]    1024 -> 1 -> Softmax
        video_size = video_input.size() # [bsz, num, dim]
        image_compressed_embeddings = video_input.view(-1, video_size[2]) # [bsz * num, dim]
        attention = self._sigmoid(self._ws1(image_compressed_embeddings)).view(video_size[0], video_size[1], -1) #[bsz, num, dim]
        attention = torch.transpose(attention, 1, 2).contiguous()  # [bsz, dim, num]
        return attention


class f_map(nn.Module): # 21 -> 1
    def __init__(self, args):
        super(f_map, self).__init__()
        self._ws1 = nn.Linear(args.f_class_dim, 1, bias=False)

        self._init_weights()
    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)


    def forward(self, video_input):
        # input is [batch_size, num, feature_dim]    21 -> 21
        video_size = video_input.size() # [bsz, num, dim]
        image_compressed_embeddings = video_input.view(-1, video_size[2]) # [bsz * num, dim]
        map = self._ws1(image_compressed_embeddings) #[bsz, num, dim]
        return map


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, args, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label)) #real_label
        self.register_buffer('fake_label', torch.tensor(target_fake_label)) #fake_label
        self.gan_mode = args.gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss() # Least Square loss in the original paper
        elif self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()  # include Sigmoid+BCELoss,  BCELoss: cross entropy
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss

class GANModel(nn.Module):
    def __init__(self, args):
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(GANModel, self).__init__()

        self._softmax = nn.Softmax(dim=-1)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_V', 'G_t', 'G_u', 'f_image', 'fg_video', 'Att_guide']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_video', 'fake_image']
        visual_names_B = ['source_image', 'source_video']

        self.visual_names = visual_names_A + visual_names_B # combine visualizations for A ,B and C
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if args.isTrain:
            self.model_names = ['G_t', 'G_u', 'Att', 'classifier', 'D_V']
        else:  # during test time, only load Gs
            self.model_names = ['G_t', 'G_u', 'Att', 'classifier']

        # define networks (both Generators and discriminators)
        self.G_t = G_t(args)
        self.G_u = G_u(args)
        self.Att = Att(args)
        self.classifier = f(args)

        if args.isTrain:
            self.D_V = D_V(args)
            # define loss functions
            self.criterionGAN = GANLoss(args)
            self.criterionCls = torch.nn.CrossEntropyLoss()
            self.criterionAtt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_t.parameters(), self.G_u.parameters(), self.Att.parameters(), \
                                                                self.classifier.parameters()), lr=args.lr, betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_V.parameters()), lr=args.lr, betas=(0.9, 0.999))

    def train_set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.real_video = input['video'].cuda()
        self.source_image = input['trimmed'].cuda()
        self.source_image_label = input['trimmed_label'].cuda()
        self.video_label = input['video_label'].cuda()


    def test_set_input(self, input):
        self.real_video = input['video'].cuda()
        self.video_label = input['video_label'].cuda()

    def forward(self):
        self.Attention = self.Att(self.real_video)
        #self.video_attention = torch.bmm(self._softmax(self.Attention), self.real_video)
        #self.fake_image = self.G_t(self.video_attention)

        self.fake_image = self.G_u(self.real_video)
        self.fake_video = self.G_t(self.source_image)

        self.real_v_logit = self.D_V(self.real_video)
        self.fake_v_logit = self.D_V(self.fake_video)

        self.video_concat = torch.cat((self.fake_image), dim=-1)
        #self.video_class_result = self.classifier(self.video_concat)
        #self.video_class_result = self.classifier(self.fake_image)
        self.video_middle_class_result = self.classifier(self.video_concat)
        self.video_class_result = torch.bmm(self.Attention, self.video_middle_class_result)/self.Attention.size(-1)


        self.image_concat = torch.cat((self.fake_video), dim=-1)
        self.image_class_result = self.classifier(self.image_concat)
        #self.image_class_result = self.classifier(self.source_image)

    def test_forward(self):
        self.Attention = self.Att(self.real_video)
        #self.video_attention = torch.bmm(self._softmax(self.Attention), self.real_video)                               #/self.Attention.size(-1)
        self.fake_image = self.G_t(self.real_video)
        self.video_concat = torch.cat((self.fake_image), dim=-1)
        self.video_middle_class_result = self.classifier(self.video_concat)
        self.video_class_result = torch.bmm(self.Attention, self.video_middle_class_result) / self.Attention.size(-1)
        #self.video_class_result = self.classifier(self.fake_image)
        self.predict_label = self._softmax(torch.squeeze(self.video_class_result, 0)).argmax(dim=1)
        return self.Attention, self.video_class_result, self.predict_label, self.video_label

    def backward_D_V(self):
        self.loss_D_V_real = self.criterionGAN(self.real_v_logit, True)
        self.loss_D_V_fake = self.criterionGAN(self.fake_v_logit, False)
        self.loss_D_V = (self.loss_D_V_real + self.loss_D_V_fake) * 0.5
        self.loss_D_V.backward(retain_graph=True)    #retain_graph=True 会保持图，但会保持参数吗?

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_att = args.lambda_att
        # video_class_result,_, _, _, _ = test_forward
        # GAN loss
        self.loss_G_t = self.criterionGAN(self.fake_i_logit, True)
        # GAN loss D_V(G_u(I))
        self.fake_v_logit = self.D_V(self.fake_video)
        self.loss_G_u = self.criterionGAN(self.fake_v_logit, True)

        # classification loss
        self.back_index = torch.tensor([20]).cuda()
        self.loss_f_image = self.criterionCls(torch.squeeze(self.image_class_result, 0), self.source_image_label) * 0.1

        self.loss_fg_video = self.criterionCls(torch.squeeze(self.video_class_result, 0), self.video_label)

        #Att guide loss
        if lambda_att > 0:
            self.video2image = self.G_u(self.real_video)
            self.image2video = self.G_t(self.source_image)
            self.videoimageconcat = torch.cat((self.video2image, self.image2video), dim=-1)
            self.clip_class = self.classifier(self.videoimageconcat)
            self.clip_class_softmax = self._softmax(torch.squeeze(self.clip_class, 0))
            self.clip_class_max = torch.max(self.clip_class_softmax, dim=1)[1]
            self.clip_num = torch.numel(self.clip_class_softmax[:,0])
            self.guide_att = torch.zeros(self.clip_num).cuda()

            for label in self.video_label:
                self.label_class_softmax = torch.squeeze(self.clip_class_softmax[:, label])
                X = torch.linspace(0, self.clip_num-1, steps=self.clip_num).cuda()
                self.fg_equal_index = torch.eq(self.clip_class_max, label).nonzero()
                self.fg_peak_num = torch.numel(self.fg_equal_index)

                Y = torch.zeros(self.clip_num).cuda()
                if self.fg_peak_num != 0:
                    for k in range(self.fg_peak_num):
                        Y = Y.float() + torch.exp(-((X.float() - self.fg_equal_index[k][0].float()).pow(2))/2)
                    Y[Y > 1.0] = 1.0
                    self.guide_att = self.guide_att + torch.mul(Y, self.label_class_softmax)
                else:
                    self.guide_att = self.guide_att + self.label_class_softmax

                Z = torch.zeros(self.clip_num).cuda()

            self.guide_att = self.guide_att/torch.numel(self.video_label)

            self.loss_att_guide = (self.criterionAtt(self.Attention[0,0,:], self.guide_att))/self.clip_num * lambda_att

        else:
            self.loss_att_guide = 0

        self.loss_G = self.loss_G_u + self.loss_G_t + self.loss_f_image + self.loss_fg_video + self.loss_att_guide
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake image/video and reconstruction image/video

        # D_A
        self.set_requires_grad([self.D_V], True)
        self.set_requires_grad([self.G_t, self.G_u, self.Att, self.classifier], False)
        self.optimizer_D.zero_grad()  # set D_V's gradients to zero
        self.backward_D_V()  # calculate graidents for D_V
        self.optimizer_D.step()  # update D_A's weights

        # G_A and G_B
        self.set_requires_grad([self.D_V], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.G_t, self.G_u, self.Att, self.classifier], True)
        self.optimizer_G.zero_grad()  # set G_t,G_u,Att,classifier's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

    def return_loss(self):
        return self.loss_D_V, self.loss_G_u, self.loss_G_t, self.loss_f_image, self.loss_fg_video, self.loss_att_guide



