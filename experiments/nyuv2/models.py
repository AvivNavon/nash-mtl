from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SegNet(nn.Module):
    """SegNet MTAN"""

    def __init__(self):
        super(_SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filter[i], filter[i]]),
                        self.conv_layer([filter[i], filter[i]]),
                    )
                )

        # define task attention layers
        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]
        )
        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])]
        )
        self.encoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[1]])]
        )
        self.decoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[0]])]
        )

        for j in range(3):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])
                )
                self.decoder_att.append(
                    nn.ModuleList(
                        [self.att_layer([2 * filter[0], filter[0], filter[0]])]
                    )
                )
            for i in range(4):
                self.encoder_att[j].append(
                    self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]])
                )
                self.decoder_att[j].append(
                    self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]])
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 2]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i]])
                )
            else:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [
            self.encoder_block,
            self.decoder_block,
            self.conv_block_enc,
            self.conv_block_dec,
            # self.encoder_att, self.decoder_att,
            self.encoder_block_att,
            self.decoder_block_att,
            self.down_sampling,
            self.up_sampling,
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[0],
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=1,
                    padding=0,
                ),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channel[0],
                out_channels=channel[1],
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channel[1],
                out_channels=channel[2],
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1]
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1)
                    )
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1]
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_encoder[i][-1][-1],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_decoder[i][j - 1][2],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return (
            [t1_pred, t2_pred, t3_pred],
            (
                atten_decoder[0][-1][-1],
                atten_decoder[1][-1][-1],
                atten_decoder[2][-1][-1],
            ),
        )


class SegNetMtan(nn.Module):
    def __init__(self):
        super().__init__()
        self.segnet = _SegNet()

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" not in n)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" in n)

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return []

    def forward(self, x, return_representation=False):
        if return_representation:
            return self.segnet(x)
        else:
            pred, rep = self.segnet(x)
            return pred


class SegNetSplit(nn.Module):
    def __init__(self, model_type="standard"):
        super(SegNetSplit, self).__init__()
        # initialise network parameters
        assert model_type in ["standard", "wide", "deep"]
        self.model_type = model_type
        if self.model_type == "wide":
            filter = [64, 128, 256, 512, 1024]
        else:
            filter = [64, 128, 256, 512, 512]

        self.class_nb = 13

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filter[i], filter[i]]),
                        self.conv_layer([filter[i], filter[i]]),
                    )
                )

        # define task specific layers
        self.pred_task1 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1
            ),
            nn.Conv2d(
                in_channels=filter[0],
                out_channels=self.class_nb,
                kernel_size=1,
                padding=0,
            ),
        )
        self.pred_task2 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1
            ),
            nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0),
        )
        self.pred_task3 = nn.Sequential(
            nn.Conv2d(
                in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1
            ),
            nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0),
        )

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # define convolutional block
    def conv_layer(self, channel):
        if self.model_type == "deep":
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=channel[1],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # global shared encoder-decoder network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(g_decoder[i][1]), dim=1)
        t2_pred = self.pred_task2(g_decoder[i][1])
        t3_pred = self.pred_task3(g_decoder[i][1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], g_decoder[i][
            1
        ]  # NOTE: last element is representation


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.segnet = SegNetSplit()

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" not in n)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.segnet.named_parameters() if "pred" in n)

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return self.segnet.conv_block_dec[-5].parameters()

    def forward(self, x, return_representation=False):
        if return_representation:
            return self.segnet(x)
        else:
            pred, rep = self.segnet(x)
            return pred
