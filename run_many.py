#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from lib import flowlib as fl
import glob
from natsort import natsorted

import xlwt



try:
    from correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation')
    import correlation  # you should consider upgrading python
# end

##########################################################

assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 41)  # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.cuda.device(1)  # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'
arguments_strFirst = './images/first.jpg'
arguments_strSecond = './images/second.jpg'
arguments_strOut = './out.flo'

for strOption, strArgument in \
        getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################

Backward_tensorGrid = {}
Backward_tensorPartial = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
    # end

    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones(
            [tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)])
    # end

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    tensorInput = torch.cat([tensorInput, Backward_tensorPartial[str(tensorFlow.size())]], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(
            Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear',
                                                   padding_mode='zeros')

    tensorMask = tensorOutput[:, -1:, :, :];
    tensorMask[tensorMask > 0.999] = 1.0;
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask


# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]

        # end

        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = \
                    [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                        intLevel + 1]
                intCurrent = \
                    [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                        intLevel + 0]

                if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2,
                                                                              kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(
                    in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2,
                    padding=1)
                if intLevel < 6: self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3,
                                    stride=1, padding=1)
                )

            # end

            def forward(self, tensorFirst, tensorSecond, objectPrevious):
                tensorFlow = None
                tensorFeat = None

                if objectPrevious is None:
                    tensorFlow = None
                    tensorFeat = None

                    tensorVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond),
                        negative_slope=0.1, inplace=False)

                    tensorFeat = torch.cat([tensorVolume], 1)

                elif objectPrevious is not None:
                    tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
                    tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

                    tensorVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tensorFirst=tensorFirst,
                                                              tensorSecond=Backward(tensorInput=tensorSecond,
                                                                                    tensorFlow=tensorFlow * self.dblBackward)),
                        negative_slope=0.1, inplace=False)

                    tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

                # end

                tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }

        # end

        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleMain(tensorInput)

        # end

        # end

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

        self.load_state_dict(torch.load('./model/network-' + arguments_strModel + '.pytorch'))

    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])


# end


# end

moduleNetwork = Network().cuda().eval()


##########################################################

def estimate(tensorFirst, tensorSecond):
    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    # assert (
    #             intWidth == 1024)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert (
    #             intHeight == 436)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst,
                                                              size=(intPreprocessedHeight, intPreprocessedWidth),
                                                              mode='bilinear', align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond,
                                                               size=(intPreprocessedHeight, intPreprocessedWidth),
                                                               mode='bilinear', align_corners=False)

    tensorFlow = 20.0 * torch.nn.functional.interpolate(
        input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth),
        mode='bilinear', align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tensorFlow[0, :, :, :].cpu()


def generate_flow_data(path1, path2):
    img1 = PIL.Image.open(path1)
    img2 = PIL.Image.open(path2)

    tensorFirst = torch.FloatTensor(
        numpy.array(img1)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0))
    tensorSecond = torch.FloatTensor(
        numpy.array(img2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0))

    tensorOutput = estimate(tensorFirst, tensorSecond).numpy().transpose(1, 2, 0)
    return tensorOutput


# end

##########################################################
if __name__ == '__main__':

    base_dir = 'D:/py/data'
    save_dir = 'D:/py/flo_data'

    game_list = os.listdir(base_dir)
    for game in game_list:
        event_list = os.listdir(os.path.join(base_dir, game))
        for event in event_list:
            print(event)
            workbook = xlwt.Workbook(encoding='utf-8')
            time_list = os.listdir(os.path.join(base_dir, game, event))
            for time in time_list:
                img_list = natsorted(glob.glob(os.path.join(base_dir, game, event, time) + '/*.jpg'))
                img_num = len(img_list)
                # if img_num % 2 == 0:
                #     end = img_num - 2
                # else:
                #     end = img_num - 1
                end = img_num - 1

                firstfilename = os.path.basename(img_list[0])
                flo_count = int(firstfilename.replace(".jpg", "")) + 1

                save_count = 1

                worksheet = workbook.add_sheet('{}'.format(time))
                worksheet.write(0, 0, label=u'time')
                worksheet.write(0, 1, label=u'maxrad')
                worksheet.write(0, 2, label=u'minu')
                worksheet.write(0, 3, label=u'maxu')
                worksheet.write(0, 4, label=u'minv')
                worksheet.write(0, 5, label=u'maxv')

                for index in range(0, end, 1):

                    img1_path = img_list[index]
                    img2_path = img_list[index + 1]

                    flo_data = generate_flow_data(img1_path, img2_path)

                    save_path = os.path.join(save_dir, game, event, time)
                    isExists = os.path.exists(save_path)
                    if not isExists:
                        os.makedirs(save_path)

                    # save orginal 2-channel flow data

                    with open(save_path + '/{}.flo'.format(flo_count), 'wb') as objectOutput:
                        numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objectOutput)
                        numpy.array([flo_data.shape[1], flo_data.shape[0]], dtype=numpy.int32).tofile(objectOutput)
                        numpy.array(flo_data, dtype=numpy.float32).tofile(objectOutput)

                    # save color-coding image
                    # jpgsave_path = os.path.join(save_path, 'jpgsave')
                    # isExists_2 = os.path.exists(jpgsave_path)
                    # if not isExists_2:
                    #     os.makedirs(jpgsave_path)
                    maxrad1, minu1, maxu1, minv1, maxv1 = fl.flow_to_image(flo_data)

                    worksheet.write(save_count, 0, flo_count)
                    worksheet.write(save_count, 1, float(maxrad1))
                    worksheet.write(save_count, 2, float(minu1))
                    worksheet.write(save_count, 3, float(maxu1))
                    worksheet.write(save_count, 4, float(minv1))
                    worksheet.write(save_count, 5, float(maxv1))

                    flo_count += 1
                    save_count += 1

                xls_save_path = os.path.join(save_dir, game, event)
                workbook.save(xls_save_path + '/mix_{}.xls'.format(event))  # 正式存的时候用
