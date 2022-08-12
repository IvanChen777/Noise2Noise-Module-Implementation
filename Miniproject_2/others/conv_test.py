import torch
import sys
import importlib
sys.path.append("/Applications/epfl/epfl2/DL/EE-559-Final-Project-main")
from Miniproject_2.model import Conv2d, NearestUpsampling, ReLU, Sigmoid, MSELoss, Sequential, SGD
import unittest
from torch import nn
import torch.nn.functional as F


class TestCov2d(unittest.TestCase):

    def test_forward(self):
        in_channels = 3
        out_channels = 4
        kernel_size = (2, 3)
        stride = (2, 3)
        conv = Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=True)

        batch_size = 3
        w = 8
        h = 8
        input = torch.randn((batch_size, in_channels, w, h))
        output = conv.forward(input)
        expected = F.conv2d(input, conv.weight, conv.bias, stride)
        self.assertTrue(torch.allclose(output, expected))

    # def test_backward(self):
    #     in_channels = 3
    #     out_channels = 8
    #     kernel_size = (2, 2)
    #     stride = 2
    #     conv = Conv2d(in_channels=in_channels,
    #                   out_channels=out_channels,
    #                   kernel_size=kernel_size,
    #                   stride=stride,
    #                   bias=True)

    #     batch_size = 3
    #     w = 8
    #     h = 8
    #     input = torch.randn((batch_size, in_channels, w, h))
    #     output = conv.forward(input)

    #     out_grad = torch.randn(output.shape)
    #     in_grad, weight_grad, bias_grad = conv.backward(out_grad)

    #     # self.assertEqual(in_grad.shape, input.shape)
    #     self.assertEqual(weight_grad.shape, (out_channels, in_channels, kernel_size[0], kernel_size[1]))
    #     self.assertEqual(bias_grad.shape, (1, out_channels))

    def test_backward(self):
        in_channels = 3
        out_channels = 2
        kernel_size = (2, 2)
        stride = 1
        conv = Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=True)

        conv.weight = torch.Tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
                                    [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]]])

        input = torch.Tensor([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                               [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                               [[18, 19, 20], [21, 22, 23], [24, 25, 26]]],
                              [[[27, 28, 29], [30, 31, 32], [33, 34, 35]],
                               [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
                               [[45, 46, 47], [48, 49, 50], [51, 52, 53]]]])

        output = conv.forward(input)

        out_grad = torch.Tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                                 [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]])
        in_grad = conv.backward(out_grad)
        print(in_grad)
        # self.assertEqual(weight_grad.shape, (out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # self.assertEqual(bias_grad.shape, (1, out_channels))
        self.assertTrue(torch.allclose(in_grad, torch.Tensor([[[[48., 112., 66.],
                                                                [128., 296., 172.],
                                                                [88., 200., 114.]],
                                                               [[64., 152., 90.],
                                                                [176., 408., 236.],
                                                                [120., 272., 154.]],
                                                               [[80., 192., 114.],
                                                                [224., 520., 300.],
                                                                [152., 344., 194.]]],
                                                              [[[144., 320., 178.],
                                                                [352., 776., 428.],
                                                                [216., 472., 258.]],
                                                               [[224., 488., 266.],
                                                                [528., 1144., 620.],
                                                                [312., 672., 362.]],
                                                               [[304., 656., 354.],
                                                                [704., 1512., 812.],
                                                                [408., 872., 466.]]]])))
    # def test_backward(self):
    #     in_channels = 3
    #     out_channels = 8
    #     kernel_size = (2, 2)
    #     stride = 2
    #     conv = Conv2d(in_channels=in_channels,
    #                 out_channels=out_channels,
    #                 kernel_size=kernel_size,
    #                 stride=stride,
    #                 bias=True)

    #     batch_size = 3
    #     w = 8
    #     h = 8
    #     input = torch.randn((batch_size, in_channels, w, h))
    #     output = conv.forward(input)

    #     out_grad = torch.randn(output.shape)
    #     in_grad = conv.backward(out_grad)
    #     print(in_grad.shape, input.shape)
    #     self.assertEqual(in_grad.shape, input.shape)
    #     self.assertEqual(conv.weight_grad.shape, (out_channels, in_channels, kernel_size[0], kernel_size[1]))
    #     self.assertEqual(conv.bias_grad.shape, (1, out_channels))

class TestUpsampling(unittest.TestCase):
    def test_forward(self):
        input = torch.Tensor([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                               [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                               [[18, 19, 20], [21, 22, 23], [24, 25, 26]]],
                              [[[27, 28, 29], [30, 31, 32], [33, 34, 35]],
                               [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
                               [[45, 46, 47], [48, 49, 50], [51, 52, 53]]]])

        upsampling = NearestUpsampling(2)

        in_channels = 3
        out_channels = 2
        kernel_size = (2, 2)
        stride = 1
        conv = Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=True)

        output = conv.forward(upsampling.forward(input))

        upsample = nn.Upsample(scale_factor=2, mode='nearest')

        expected = F.conv2d(upsample(input), conv.weight, conv.bias, stride)

        self.assertTrue(torch.allclose(output, expected))


class TestReLU(unittest.TestCase):
    def test_forward(self):
        input = torch.Tensor([[[[-0, 1, 2], [3, -4, 5], [6, 7, -8]],
                               [[-9, 10, 11], [12, -13, 14], [15, 16, -17]],
                               [[-18, 19, 20], [21, -22, 23], [24, 25, -26]]],
                              [[[-27, 28, 29], [30, -31, 32], [33, 34, -35]],
                               [[-36, 37, 38], [39, -40, 41], [42, 43, -44]],
                               [[-45, 46, 47], [48, -49, 50], [51, 52, -53]]]])

        relu = ReLU()

        output = relu.forward(input)

        self.assertTrue(torch.allclose(output, F.ReLU(input)))

        grad = torch.Tensor([[[[-0, 1, 2], [3, -4, 5], [6, 7, -8]],
                              [[-9, 10, 11], [12, -13, 14], [15, 16, -17]],
                              [[-18, 19, 20], [21, -22, 23], [24, 25, -26]]],
                             [[[-27, 28, 29], [30, -31, 32], [33, 34, -35]],
                              [[-36, 37, 38], [39, -40, 41], [42, 43, -44]],
                              [[-45, 46, 47], [48, -49, 50], [51, 52, -53]]]])

        in_grad = relu.backward(grad)

        print((in_grad))


class TestSigmoid(unittest.TestCase):
    def test_forward(self):
        input = torch.Tensor([[[[-0, 1, 2], [3, -4, 5], [6, 7, -8]],
                               [[-9, 10, 11], [12, -13, 14], [15, 16, -17]],
                               [[-18, 19, 20], [21, -22, 23], [24, 25, -26]]],
                              [[[-27, 28, 29], [30, -31, 32], [33, 34, -35]],
                               [[-36, 37, 38], [39, -40, 41], [42, 43, -44]],
                               [[-45, 46, 47], [48, -49, 50], [51, 52, -53]]]])

        sig = Sigmoid()

        output = sig.forward(input)

        self.assertTrue(torch.allclose(output, F.sigmoid(input)))

        grad = torch.Tensor([[[[-0, 1, 2], [3, -4, 5], [6, 7, -8]],
                              [[-9, 10, 11], [12, -13, 14], [15, 16, -17]],
                              [[-18, 19, 20], [21, -22, 23], [24, 25, -26]]],
                             [[[-27, 28, 29], [30, -31, 32], [33, 34, -35]],
                              [[-36, 37, 38], [39, -40, 41], [42, 43, -44]],
                              [[-45, 46, 47], [48, -49, 50], [51, 52, -53]]]])

        in_grad = sig.backward(grad)

        print(in_grad)


class TestMSE(unittest.TestCase):
    def test_forward(self):
        input = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)

        loss = MSELoss()

        output = loss.forward(input, target)

        self.assertTrue(torch.allclose(output, F.mse_loss(input, target)))

        in_grad = loss.backward()


class TestSequential(unittest.TestCase):

    def test_forward(self):
        x = torch.randn(1, 3, 32, 32)

        conv = Conv2d(3, 3, 3)
        self.assertTrue(torch.allclose(conv.forward(x), F.conv2d(x, conv.weight, conv.bias)))

        sigmoid = Sigmoid()
        self.assertTrue(torch.allclose(sigmoid.forward(x), torch.sigmoid(x)))

        seq = Sequential(conv, sigmoid)
        self.assertTrue(torch.allclose(seq.forward(x), F.conv2d(x, conv.weight, conv.bias).sigmoid()))


class TestModules(unittest.TestCase):

    def test(self):
        conv1 = Conv2d(3, 3, 2, stride=2)
        conv2 = Conv2d(3, 3, 2, stride=2)
        conv3 = Conv2d(3, 3, 4)
        conv4 = Conv2d(3, 3, 8)

        conv1.weight = torch.randn(3, 3, 2, 2)
        conv2.weight = torch.randn(3, 3, 2, 2)
        conv3.weight = torch.randn(3, 3, 4, 4)
        conv4.weight = torch.randn(3, 3, 8, 8)

        seq = Sequential(conv1, ReLU(),
                         conv2, ReLU(),
                         NearestUpsampling(2), conv3, ReLU(),
                         NearestUpsampling(3), conv4, Sigmoid())

        optimizer = SGD(seq.param(), 0.1)

        loss = MSELoss()

        input = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)

        output = seq.forward(input)

        expected = []

        expected1 = F.conv2d(input, conv1.weight, conv1.bias, stride=2)
        expected1.retain_grad()
        expected.append(expected1)

        expected2 = expected1.relu()
        expected2.retain_grad()
        expected.append(expected2)

        expected3 = F.conv2d(expected2, conv2.weight, conv2.bias, stride=2)
        expected3.retain_grad()
        expected.append(expected3)

        expected4 = expected3.relu()
        expected4.retain_grad()
        expected.append(expected4)

        expected5 = F.upsample(expected4, scale_factor=2)
        expected5.retain_grad()
        expected.append(expected5)

        expected6 = F.conv2d(expected5, conv3.weight, conv3.bias)
        expected6.retain_grad()
        expected.append(expected6)

        expected7 = expected6.relu()
        expected7.retain_grad()
        expected.append(expected7)

        expected8 = F.upsample(expected7, scale_factor=3)
        expected8.retain_grad()
        expected.append(expected8)

        expected9 = F.conv2d(expected8, conv4.weight, conv4.bias)
        expected9.retain_grad()
        expected.append(expected9)

        expected10 = expected9.sigmoid()
        expected10.retain_grad()

        self.assertTrue(torch.allclose(output, expected10, rtol=1e-03, atol=1e-04))

        l = loss.forward(output, target)
        expected_l = F.mse_loss(expected10, target)
        expected_l.retain_grad()

        self.assertTrue(torch.allclose(l, expected_l))

        grad = loss.backward()
        expected_l.backward()
        self.assertTrue(torch.allclose(grad, expected10.grad, rtol=1e-05, atol=1e-05))

        self.assertEqual(len(seq.modules) - 1, len(expected))

        for i in range(len(expected) - 1, -1, -1):
            grad = seq.modules[i + 1].backward(grad)
            self.assertTrue(torch.allclose(expected[i].grad, grad, rtol=1e-03, atol=1e-04))
a=TestCov2d()
a.test_backward()
source_imgs , target_imgs = torch.load ("/Applications/epfl/epfl2/DL/EE-559-Final-Project-main/Miniproject_2/Data/train_data.pkl")
noisy_imgs , clean_imgs = torch.load ("/Applications/epfl/epfl2/DL/EE-559-Final-Project-main/Miniproject_2/Data/val_data.pkl")
#convolution
conv_1 = Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride=2)
t_conv_1 = torch.nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride=2)
t_conv_1.weight.data = conv_1.weight.clone()
t_conv_1.bias.data = conv_1.bias.clone()

conv_2 = Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride=2)
t_conv_2 = torch.nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride=2)
t_conv_2.weight.data = conv_2.weight.clone()
t_conv_2.bias.data = conv_2.bias.clone()

#input
input = source_imgs[0:1, :, :, :].type(torch.float)
print("input shape")
print(input.shape)

#model
model = Sequential(conv_1, conv_2)
torch_model = torch.nn.Sequential(t_conv_1, t_conv_2)

#forward
myoutput = model.forward(input)
torchoutput = torch_model.forward(input)
print("output shape:")
print(torchoutput.shape)

print("difference abs")
print((myoutput - torchoutput).abs().sum())


#backward
target = torch.rand(torchoutput.shape)
print("target shape:")
print(target.shape)
mse=MSELoss()
loss = mse.forward(torchoutput, target)
loss.backward()
model.backward(mse.backward())

torch.testing.assert_allclose(conv_1.param()[1][1], t_conv_1.bias.grad)
torch.testing.assert_allclose(conv_1.param()[0][1], t_conv_1.weight.grad)
