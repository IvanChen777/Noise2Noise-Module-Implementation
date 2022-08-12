
from torch import empty
from torch.nn.functional import fold, unfold
import pickle
import math
from pathlib import Path

class Module(object):
    """
    All subclasses of `Module` should implement methods `forward`, `backward` and `param`. 

    - `forward` method do calculations on the input and return the result, which will be passed to the next layer. 

    - `backward` method calculate the loss gradient w.r.t the parameters as well as the input tensor based on the gradient w.r.t the output of this layer. 

    - `param` method return the parameters of this layer. In this project, only Conv2d and Upsampling have parameters. 

    """

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=0, bias=True) -> None:
        """
        Conv2d module do convolution on the input tensors. 

        - For `forward`, we use the method mentioned in the appendix of the project description to convert the convolution operation to matrix multiplication, which speed up the computation a lot compared to using for-loop to do convolution. 

        - For `backward`, we also use the unfolded matrix mulplication to do convolution.

        - For `param`, it returns `[[weight, weight_gradient], [bias, bias_gradient]]`.

        Args:
            in_channels (int): the channel of image for the input
            out_channels (int): the channel of image for the output
            kernel_size (int/tuple): the size of kernel
            dilation (int): ingored
            stride (int, optional):  Defaults to 1.
            padding (int, optional): Defaults to 0.
            bias (bool, optional): Defaults to True.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(stride) is int:
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if type(padding) is int:
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.bias = bias
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.weight = empty((out_channels, in_channels, *self.kernel_size)
                            ).normal_()/math.sqrt((in_channels+out_channels)/2)
        self.bias = empty(out_channels).normal_() / \
            math.sqrt((in_channels+out_channels)/2)
        self.weight_grad = empty(
            (out_channels, in_channels, *self.kernel_size))
        self.bias_grad = empty(out_channels)

        self.input_size = None
        self.input_features = None

    def forward(self, input):
        """
        Args:
            input (Tensor): (batch_size, in_channel, w, h)

        Returns:
            Tensor: (batch_size, out_channel, (w + 2 * padding[0] - kernel[0]) / stride[0] + 1, (h + 2 * padding[1] - kernel[1]) / stride[1] + 1)
        """
        batch_size = input.shape[0]
        w, h = input.shape[2], input.shape[3]
        self.input_size = input.shape
        self.input_features = unfold(
            input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        wxb = self.weight.view(
            self.out_channels, -1) @ self.input_features + self.bias.view(1, -1, 1)
        return wxb.view(batch_size,
                        self.out_channels,
                        (w + 2 * self.padding[0] -
                         self.kernel_size[0]) // self.stride[0] + 1,
                        (h + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1)

    def backward(self, gradwrtoutput):
        """
        Args:
            input (Tensor): (batch_size, out_channel, (w + 2 * padding[0] - kernel[0]) / stride[0] + 1, (h + 2 * padding[1] - kernel[1]) / stride[1] + 1)

        Returns:
            Tensor: (batch_size, in_channel, w, h)
        """
        batch_size = gradwrtoutput.shape[0]
        grad = gradwrtoutput.permute(0, 2, 3, 1).view(
            (batch_size, -1, self.out_channels))
        delta_in = gradwrtoutput.permute(
            0, 2, 3, 1).reshape((-1, self.out_channels))
        self.weight_grad.add_(((self.input_features @ grad).sum(dim=0) /
                              batch_size).permute(1, 0).view(self.weight.shape))
        self.bias_grad.add_(delta_in.sum(dim=0, keepdims=False) / batch_size)
        delta_out = fold((grad @ self.weight.view(self.out_channels, -1)).permute(0, 2, 1),
                         output_size=(self.input_size[2], self.input_size[3]),
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         stride=self.stride)
        return delta_out

    def param(self):
        """ 

        Returns: 
            List: [[param, param gradient], ...]
        """
        return [[self.weight, self.weight_grad], [self.bias, self.bias_grad]]


class NNUpsampling(Module):
    """

    For `Upsampling` layer:

    - The `forward` method resize the tensor by a scale factor and interpolate the tensor with the value of the nearest element. 

    - The `backward` method, conversely, sum up the gradients of one neighbourhood into one element. 

    """

    def __init__(self, scale_factor=None) -> None:
        if type(scale_factor) is int:
            self.scale_factor = (scale_factor, scale_factor)
        else:
            self.scale_factor = scale_factor

    def forward(self, input):
        """
        Args:
            input (Tensor): (batch_size, channel, w, h)

        Returns:
            Tensor: (batch_size, channel, scale[0] * w, scale[1] * h)
        """
        return input.repeat_interleave(self.scale_factor[1], 3).repeat_interleave(self.scale_factor[0], 2)

    def backward(self, gradwrtoutput):
        """
        Args:
            input (Tensor): (batch_size, channel, scale[0] * w, scale[1] * h)

        Returns:
            Tensor: (batch_size, channel, w, h)
        """
        return gradwrtoutput.unfold(3, self.scale_factor[1], self.scale_factor[1])\
            .sum(dim=4).unfold(2, self.scale_factor[0], self.scale_factor[0]).sum(dim=4)

    def param(self):
        return []


class Upsampling(Module):
    """ 
    This layer consists of a Upsampling layer and a Conv2d layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, scale_factor=2, bias=True) -> None:
        self.upsample = NNUpsampling(scale_factor)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           dilation=dilation, stride=stride, padding=padding, bias=bias)

    def forward(self, input):
        return self.conv.forward(self.upsample.forward(input))

    def backward(self, gradwrtoutput):
        return self.upsample.backward(self.conv.backward(gradwrtoutput))

    def param(self):
        return self.conv.param()


class ReLU(Module):
    """
    - For `forward`, it returns the relu function value of the input.

    - For `backward`, it returns the gradient value.
    """

    def __init__(self) -> None:
        self.input_features = None

    def forward(self, input):
        self.input_features = input.clone()
        return input.relu()

    def backward(self, gradwrtoutput):
        return (self.input_features >= 0) * gradwrtoutput

    def param(self):
        return []


class Sigmoid(Module):
    """
    - For `forward`, it returns the sigmoid function value of the input.

    - For `backward`, it returns the gradient value.

    """

    def __init__(self) -> None:
        self.input_sig = None

    def forward(self, input):
        self.input_sig = input.sigmoid()
        return self.input_sig

    def backward(self, gradwrtoutput):
        return self.input_sig * (1 - self.input_sig) * gradwrtoutput

    def param(self):
        return []


class MSE(Module):
    """ 
    - For `forward`, given input and target, it returns the Mean Square Error loss.

    - For `backward`, it returns the gradient value.

    """

    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, input, target):
        self.input = input.clone()
        self.target = target.clone()
        return ((input - target) * (input - target)).mean()

    def backward(self):
        return 2 * (self.input - self.target) / self.input.view(-1).shape[0]

    def param(self):
        return []


class Sequential(Module):
    """
    This layer is a list of modules. 

    - For `forward`, it passes the input through the modules. 

    - For `backward`, it passes the gradient through the modules reversely. 

    - For `param`, it returns all the parameters and their gradients of the modules.

    """

    def __init__(self, *args: Module):
        self.modules = args

    def forward(self, input):
        output = input
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, gradwrtoutput):
        in_grad = gradwrtoutput
        for module in reversed(self.modules):
            in_grad = module.backward(in_grad)
        return in_grad

    def param(self):
        params = []
        for module in self.modules:
            if len(module.param()) > 0:
                params += module.param()
        return params


class SGD:
    """
    `SGD` is an optimizer. It receives the parameters of the modules when it is constructed and updates the parameters, lr is the learning rate.
    `zero_grad`: The gradients should be reset to zero every time before calling backward.
    """

    def __init__(self, params, lr, maximize=False):
        self.params = params
        self.lr = lr
        self.maximize = maximize

    def step(self):
        for param in self.params:
            if self.maximize:
                param[0] += self.lr * param[1]
            else:
                param[0] -= self.lr * param[1]

    def zero_grad(self):
        for param in self.params:
            param[1].zero_()


# For mini-project 2
class Model():
    def __init__(self) -> None:
        """ 
        instantiate model + optimizer + loss function 

        """
        channel = 42
        conv1 = Conv2d(in_channels=3, out_channels=channel,
                       kernel_size=2, dilation=1, stride=2, padding=0, bias=True)
        conv2 = Conv2d(in_channels=channel, out_channels=channel,
                       kernel_size=2, dilation=1, stride=2, padding=0, bias=True)
        upsampling1 = Upsampling(scale_factor=2, in_channels=channel, out_channels=channel,
                                 kernel_size=3, dilation=1, stride=1, padding=1, bias=True)
        upsampling2 = Upsampling(scale_factor=2, in_channels=channel, out_channels=3,
                                 kernel_size=3, dilation=1, stride=1, padding=1, bias=True)

        self.layer = Sequential(conv1, ReLU(),
                                conv2, ReLU(),
                                upsampling1,  ReLU(),
                                upsampling2, Sigmoid())
        self.opt = SGD(self.layer.param(), 4)
        self.loss = MSE()
        self.batch_size = 4

    def load_pretrained_model(self) -> None:
        """
        This loads the parameters saved in bestmodel.pth into the model pass


        """

        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, "rb") as f:
            params = pickle.load(f)
        for i, param in enumerate(params):
            if len(param) > 0:
                if isinstance(self.layer.modules[i], Conv2d):
                    self.layer.modules[i].weight = param[0][0]
                    self.layer.modules[i].bias = param[1][0]
                elif isinstance(self.layer.modules[i], Upsampling):
                    self.layer.modules[i].conv.weight = param[0][0]
                    self.layer.modules[i].conv.bias = param[1][0]

    def forward(self, x):
        """

        Args:
            x (Tensor): Test_input

        Returns:
            Tensor: the tenor after foward function based on the sequential layer
        """
        x = self.layer.forward(x)
        return x

    def train(self, train_input, train_target, num_epochs) -> None:
        """Building our own training function

        Args:
            train_input (Tensor): tensor of size (N, C, H, W) containing a noisy version of the images
            train_target (Tensor):tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise
            num_epochs (int): the number of epochs
        """

        train_input = train_input.float() / 255.  # set the input to [0,1]
        train_target = train_target.float() / 255.  # set the target to [0,1]

        for e in range(num_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), self.batch_size):
                self.opt.zero_grad()
                output = self.forward(
                    train_input.narrow(0, b, self.batch_size))
                Loss = self.loss.forward(
                    output, train_target.narrow(0, b, self.batch_size))
                acc_loss += Loss
                loss_grad = self.loss.backward()
                self.layer.backward(loss_grad)
                self.opt.step()
                # After training 10000 images, print the loss
                if b % 10000 == 9999:
                    print('training acc_loss is '+str(acc_loss / 10000))
                    acc_loss = 0.
        # save the best parameters by pickle
        params = []
        for module in self.layer.modules:
            params.append(module.param())
        
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, "wb") as f:
            pickle.dump(params, f)

    def predict(self, test_input) -> None:
        """Building our own predict function

        Args:
            test_input (Tensor): tensor of size (N1, C, H, W) with values in range 0-255 that has to be denoised by the trained or the loaded network.

        Returns:
            Tensor: a tensor of the size (N1, C, H, W) with values in range 0-255. 
        """
        test_input = test_input.float() / 255.
        x = self.forward(test_input)
        return x*255.
