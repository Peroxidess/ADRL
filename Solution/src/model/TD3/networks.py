import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc2_dim//2)
        self.fc4 = nn.Linear(fc2_dim//2, fc2_dim//4)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        action = self.action(x)
        return action

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file, map_location=T.device('cpu')))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc2_dim//2)
        self.fc4 = nn.Linear(fc2_dim//2, fc2_dim//4)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file, map_location=T.device('cpu')))


class ClassifyNetwork(nn.Module):
    def __init__(self, gama, state_dim, action_dim, fc1_dim, fc2_dim, num_class=2):
        super(ClassifyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim-1, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc2_dim//2)
        self.sfm = nn.Softmax()
        self.sm = nn.Sigmoid()
        self.y = nn.Linear(fc2_dim//2, num_class)
        self.flag_nec = True
        self.num_proto = 2
        self.num_classes = num_class
        self.protos = nn.Parameter(T.randn(self.num_proto * self.num_classes, fc2_dim//2), requires_grad=True)
        self.radius = nn.Parameter(T.rand(1, self.num_proto * self.num_classes) - 0.4, requires_grad=True)
        self.optimizer = optim.Adam(self.parameters(), lr=gama, weight_decay=1e-3)
        self.loss = FocalLoss(num_class, size_average=True)
        self.to(device)

    def forward(self, x):
        x_ff = F.gelu(self.fc1(x))
        x_ff = F.gelu(self.fc2(x_ff))
        x_ff = self.fc3(x_ff)
        if self.flag_nec:
            dist = self.nce_prob_euc(x_ff)
            prob = dist
            pass
        else:
            y = self.y(x_ff)
            prob = self.sfm(y)
            dist = prob
            pass
        return prob, dist

    def nce_prob_cos(self, feat):
        dist = self.cosine_distance_func(feat, self.protos)
        dist = (dist / self.radius.sigmoid()).sigmoid()
        cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(1)
        return cls_score

    def nce_prob_euc(self, feat):
        dist = self.euclidean_distance_func(feat.sigmoid(), self.protos.sigmoid())
        dist = T.exp(-(dist ** 2) / (2 * self.radius.sigmoid() ** 2))
        cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(1)
        return cls_score

    '''
    Reference:
    https://github.com/WanFang13/NCE-Net
    '''
    @staticmethod
    def normed_euclidean_distance_func(feat1, feat2):
        # Normalized Euclidean Distance
        # feat1: N * Dim
        # feat2: M * Dim
        # out:   N * M Euclidean Distance
        feat1, feat2 = F.normalize(feat1), F.normalize(feat2)
        feat_matmul = T.matmul(feat1, feat2.t())
        distance = T.ones_like(feat_matmul) - feat_matmul
        distance = distance * 2
        return distance.clamp(1e-10).sqrt()

    @staticmethod
    def euclidean_distance_func(feat1, feat2):
        # Euclidean Distance
        # feat1: N * Dim
        # feat2: M * Dim
        # out:   N * M Euclidean Distance
        feat1_square = T.sum(T.pow(feat1, 2), 1, keepdim=True)
        feat2_square = T.sum(T.pow(feat2, 2), 1, keepdim=True)
        feat_matmul = T.matmul(feat1, feat2.t())
        distance = feat1_square + feat2_square.t() - 2 * feat_matmul
        return distance.clamp(1e-10).sqrt()

    @staticmethod
    def cosine_distance_func(feat1, feat2):
        # feat1: N * Dim
        # feat2: M * Dim
        # out:   N * M Cosine Distance
        distance = T.matmul(F.normalize(feat1), F.normalize(feat2).t())
        return distance

    @staticmethod
    def cosine_distance_full_func(feat1, feat2):
        # feat1: N * Dim
        # feat2: M * Dim
        # out:   (N+M) * (N+M) Cosine Distance
        feat = T.cat((feat1, feat2), dim=0)
        distance = T.matmul(F.normalize(feat), F.normalize(feat).t())
        return distance


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(T.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = inputs

        ids = targets.view(-1, 1)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        probs_pos = P[:, 1].reshape(-1, 1)
        probs_neg = P[:, 0].reshape(-1, 1)

        batch_loss = - (T.pow((1 - probs_neg), self.gamma) * probs_neg.log() + T.pow((probs_pos), self.gamma) * (1 - probs_pos).log()) * ids - \
                     (T.pow(probs_neg, self.gamma) * (1 - probs_neg).log() + T.pow((1 - probs_pos), self.gamma) * probs_pos.log()) * (1 - ids) # label 0
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss