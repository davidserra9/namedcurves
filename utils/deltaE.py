from skimage import color
import numpy as np
import torch

class deltaEab():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area

    def __call__(self, img1, img2):
        """ Compute the deltaE76 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE.py
        :param img1: numpy RGB image or pytorch tensor
        :param img2: numpy RGB image or pytocrh tensor
        :return: deltaE76
        """

        if type(img1) == torch.Tensor:
            assert img1.shape[0] == 1
            img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()

        if type(img2) == torch.Tensor:
            assert img2.shape[0] == 1
            img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE76
        de76 = np.sqrt(np.sum(np.power(img1 - img2, 2), 1))

        return sum(de76) / (np.shape(de76)[0] - self.color_chart_area)


class deltaE00():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area
        self.kl = 1
        self.kc = 1
        self.kh = 1

    def __call__(self, img1, img2):
        """ Compute the deltaE00 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE2000.py
        :param img1: numpy RGB image or pytocrh tensor
        :param img2: numpy RGB image or pytocrh tensor
        :return: deltaE00
        """

        if type(img1) == torch.Tensor:
            assert img1.shape[0] == 1
            img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()

        if type(img2) == torch.Tensor:
            assert img2.shape[0] == 1
            img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE00
        Lstd = np.transpose(img1[:, 0])
        astd = np.transpose(img1[:, 1])
        bstd = np.transpose(img1[:, 2])
        Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
        Lsample = np.transpose(img2[:, 0])
        asample = np.transpose(img2[:, 1])
        bsample = np.transpose(img2[:, 2])
        Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
        Cabarithmean = (Cabstd + Cabsample) / 2
        G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
            Cabarithmean, 7) + np.power(25, 7))))
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
        Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
        Cpprod = (Cpsample * Cpstd)
        zcidx = np.argwhere(Cpprod == 0)
        hpstd = np.arctan2(bstd, apstd)
        hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
        hpsample = np.arctan2(bsample, apsample)
        hpsample = hpsample + 2 * np.pi * (hpsample < 0)
        hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
        dL = (Lsample - Lstd)
        dC = (Cpsample - Cpstd)
        dhp = (hpsample - hpstd)
        dhp = dhp - 2 * np.pi * (dhp > np.pi)
        dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
        dhp[zcidx] = 0
        dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp = hp + (hp < 0) * 2 * np.pi
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
        Lpm502 = np.power((Lp - 50), 2)
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
            0.32 * np.cos(3 * hp + np.pi / 30) \
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            - np.power((180 / np.pi * hp - 275) / 25, 2))
        Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
        RT = - np.sin(2 * delthetarad) * Rc
        klSl = self.kl * Sl
        kcSc = self.kc * Sc
        khSh = self.kh * Sh
        de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                       np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))

        return np.sum(de00) / (np.shape(de00)[0] - self.color_chart_area)
