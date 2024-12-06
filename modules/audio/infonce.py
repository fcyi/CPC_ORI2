"""
InfoNCE
Calculates the 'Info Noise-Contrastive-Estimation' as explained by Van den Oord et al. (2018),
implementation by Bas Veeling & Sindy Lowe
"""

import torch
import torch.nn as nn
import numpy as np


class InfoNCE(nn.Module):
    def __init__(self, args, gar_hidden, genc_hidden):
        super(InfoNCE, self).__init__()

        self.args = args
        self.gar_hidden = gar_hidden
        self.genc_hidden = genc_hidden
        self.negative_samples = self.args.negative_samples

        # predict |prediction_step| timesteps into the future
        self.predictor = nn.Linear(
            gar_hidden, genc_hidden * self.args.prediction_step, bias=False
        )

        if self.args.subsample:
            self.subsample_win = 128

        self.loss = nn.LogSoftmax(dim=1)

    def get(self, x, z, c):
        full_z = z

        if self.args.subsample:
            """ 
            positive samples are restricted to this subwindow to reduce the number of calculations for the loss, 
            negative samples can still come from any point of the input sequence (full_z)
            """
            if c.size(1) > self.subsample_win:
                # 对于给定批次中的每一条语段，皆选取若干个时刻作为正样本
                # 获取这些语段在这些位置处的上下文表征以及真实编码
                seq_begin = np.random.randint(0, c.size(1) - self.subsample_win)
                c = c[:, seq_begin : seq_begin + self.subsample_win, :]
                z = z[:, seq_begin : seq_begin + self.subsample_win, :]
        # 产生抽取到的批次中的每一条语段中的若干个时刻的上下文表征相应时刻之后若干个时刻的预测编码
        # 在此处Wc[i, j, (k-1)*encDim:k*encDim]表示的是该批数中，第i个语段在第j+k-1+1 = j+k的预测编码
        Wc = self.predictor(c)
        return self.infonce_loss(Wc, z, full_z)

    def broadcast_batch_length(self, input_tensor):
        """
        broadcasts the given tensor in a consistent way, such that it can be applied to different inputs and
        keep their indexing compatible
        :param input_tensor: tensor to be broadcasted, generally of shape B x L x C
        :return: reshaped tensor of shape (B*L) x C
        """

        assert input_tensor.size(0)
        assert len(input_tensor.size()) == 3

        # 将批次维度和时刻维度混在一起，即(B,L,C) -> (B*L,C)
        return input_tensor.reshape(-1, input_tensor.size(2))

    def get_pos_sample_f(self, Wc_k, z_k):
        """
        calculate the output of the log-bilinear model for the positive samples, i.e. where z_k is the actual
        encoded future that had to be predicted
        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)  # (B*(L-K), C) -> (B*(L-K), 1, C)
        z_k = z_k.unsqueeze(2)  # (B*(L-K), C) ->　(B*(L-K), C，1)
        f_k = torch.squeeze(torch.matmul(Wc_k, z_k), 1)  # (B*(L-K), 1)
        return f_k

    def get_neg_z(self, z):
        """
        scramble z to retrieve negative samples, i.e. z values that should not be predicted by the model
        :param z: unshuffled z as output by the model
        :return: z_neg - shuffled z to be used for negative sampling
                shuffling params rand_neg_idx, rand_offset for testing this function
        """

        """ randomly selecting from all z values; 
            can cause positive samples to be selected as negative samples as well 
            (but probability is <0.1% in our experiments)
            done once for all time-steps, much faster  
            torch.index_select(input, dim, index, *, out=None) → Tensor
            选择根据给定的index在input中沿dim维度进行索引选择张量数据，相当于更高级的索引功能。
            返回的张量数组与input具有相同的维数，这里与直接进行索引有区别；
            返回的张量数据在dim维度上的尺寸大小与index的长度相同，其他尺寸大小与原始张量中的尺寸相同；
            index：维数必须小于等于1；
            torch.randperm(n) 函数用于生成一个从 0 到 n-1 的随机排列的整数序列。常用于随机打乱数据或索引时，
        """
        z = self.broadcast_batch_length(z)
        # 用于将多个张量按照指定的维度进行堆叠。
        # 它接受一个可迭代对象作为输入，其中的每个元素都是一个张量，
        # 并且新增一个dim维度，然后将这些张量在该维度上进行堆叠。
        z_neg = torch.stack(
            [
                torch.index_select(z, 0, torch.randperm(z.size(0)).to(z.get_device()))
                for i in range(self.negative_samples)
            ],
            2,
        )  # (B*L, C) -> (B*L, C, self.negative_samples)
        rand_neg_idx = None
        rand_offset = None
        return z_neg, rand_neg_idx, rand_offset

    def get_neg_samples_f(self, Wc_k, z_k, z_neg=None, k=None):
        """
        calculate the output of the log-bilinear model for the negative samples. For this, we get z_k_neg from z_k
        by randomly shuffling the indices.
        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)  # (B*(L-K), C) -> (B*(L-K), 1, C)

        """
            by shortening z_neg from the front, we get different negative samples
            for every prediction-step without having to re-sample;
            this might cause some correlation between the losses within a batch
            (e.g. negative samples for projecting from z_t to z_(t+k+1) 
            and from z_(t+1) to z_(t+k) are the same)                
        """

        # B*L-B*(L-K) = B*K
        # B*L - B*K = B*(L-K)
        # z_neg[B*K:, :, :]: (B*(L-k), C, self.negative_samples)
        z_k_neg = z_neg[z_neg.size(0) - Wc_k.size(0) :, :, :]
        # (B*(L-K), 1, C), (B*(L-K), C, self.negative_samples) -> (B*(L-K), self.negative_samples)
        f_k = torch.squeeze(torch.matmul(Wc_k, z_k_neg), 1)

        return f_k

    def infonce_loss(self, Wc, z, full_z):
        """
        calculate the loss based on the model outputs Wc (the prediction) and z (the encoded future)
        :param Wc: output of the predictor, where W are the weights for the different timesteps and
        c the latent representation, Wc:  (B, L, C * self.args.prediction_step)
        :param z: encoded future - output of the encoder (B, L, C)
        :return: loss - average loss over all samples, timesteps and prediction steps in the batch
                accuracy - average accuracies over all samples, timesteps and predictions steps in the batch
        """
        seq_len = z.size(1)
        total_loss = 0
        accuracies = torch.zeros(self.args.prediction_step, 1)
        # 真实标签，因为在构建正负样本时统一将正样本放在第一个位置，
        # 所以真实标签（即正样本所在的位置）皆为0
        true_labels = torch.zeros((seq_len * self.args.batch_size,)).long()

        # Which type of method to use for negative sampling:
        # 0 - inside the loop for the prediction time-steps. Slow, but samples from all but the current pos sample
        # 1 - outside the loop for prediction time-steps
        #   Low probability (<0.1%，可以参考错排问题) of sampling the positive sample as well.
        # 2 - outside the loop for prediction time-steps. Sampling only within the current sequence
        #   Low probability of sampling the positive sample as well.

        # sampling method 1 / 2
        z_neg, _, _ = self.get_neg_z(full_z)

        # 此处的prediction_step对应预测时刻，但是真实编码和预测编码的关系构建却有点复杂
        # assume: c: [B, L, C2], z: [B, L, C1], Wc: [B, L, ts*C1]
        # c[i, j, C2]表示第i个语段中第j个时刻下的全局表征
        # z[i, j, C1]表示第i个语段中第j个时刻下的编码表征
        # 这也就意味着Wc[i, t, (k-1)*encDim:k*encDim]为第i个语段中第t+k时刻下的预测编码表征
        # 换而言之，
        # z[i, 1, 0:C1] 对应着 Wc[i, 0, :C1]
        # z[i, 2, 0:C1] 对应着 Wc[i, 1, :C1], Wc[i, 0, C1:2*C1]
        # z[i, 3, 0:C1] 对应着 Wc[i, 2, :C1], Wc[i, 1, C1:2*C1], Wc[i, 0, 2*C1:3*C1]
        # ...
        # z[i, n, 0:C1] 对应着 Wc[i, n-1, :C1], Wc[i, n-2, C1:2*C1], ..., Wc[i, 1, (n-2)*C1:(n-1)*C1], Wc[i, 0, (n-1)*C1:n*C1]
        # 由此，就可以得到以下对应关系：
        # z[:B, k:, :]对应着Wc[:B, :-k, (k-1)*encDim:k*encDim]
        # 此处的做法，能够CPC对于不同时刻的编码表征预测能力训练所用的样本，对于第k个时刻，用来训练的样本多达B*(L-k)个，显然，k越小，用于训练的样本越多
        for k in range(1, self.args.prediction_step + 1):
            # z_k是采样窗口中的后L-K个，Wc_k是采样窗口的前L-K个
            z_k = z[:, k:, :]  # (B, L-K, C)
            Wc_k = Wc[:, :-k, (k - 1) * self.genc_hidden : k * self.genc_hidden]  # (B, L-K, C)

            z_k = self.broadcast_batch_length(z_k)  # (B, L-K, C) -> (B*(L-K), C)
            Wc_k = self.broadcast_batch_length(Wc_k)  # (B, L-K, C) -> (B*(L-K), C)

            pos_samples = self.get_pos_sample_f(Wc_k, z_k)  # (B*(L-K), 1)

            neg_samples = self.get_neg_samples_f(Wc_k, z_k, z_neg, k)  # (B*(L-K), self.negative_samples)

            # concatenate positive and negative samples
            # # (B*(L-K), 1+self.negative_samples)
            results = torch.cat((pos_samples, neg_samples), 1)
            # 沿着行方向进行log_softmax，并且获取其中正样本的预测情况。
            # 由于引入softmax进行归一化，所以仅基于正样来构建损失时是可以的
            loss = self.loss(results)[:, 0]

            total_samples = (seq_len - k) * self.args.batch_size
            loss = -loss.sum() / total_samples
            total_loss += loss

            # calculate accuracy
            if self.args.calc_accuracy:
                predicted = torch.argmax(results, 1)
                # # 预测正确的正样本数目
                correct = (
                    (predicted == true_labels[: (seq_len - k) * self.args.batch_size])
                    .sum()
                    .item()
                )
                accuracies[k - 1] = correct / total_samples  # 记录对第k个时刻的预测精度

        total_loss /= self.args.prediction_step
        accuracies = torch.mean(accuracies)

        return total_loss, accuracies

