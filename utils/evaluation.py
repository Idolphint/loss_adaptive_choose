import numpy as np
import xlrd
from scipy.io import loadmat



class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        # return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc[0], classAcc[1]  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # åå¯¹è§åç´ çå¼ï¼è¿ååè¡¨
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1è¡¨ç¤ºæ··æ·ç©éµè¡çå¼ï¼è¿ååè¡¨ï¼?axis = 0è¡¨ç¤ºåæ··æ·ç©éµåçå¼ï¼è¿ååè¡¨
        IoU = intersection / union  # è¿ååè¡¨ï¼å¶å¼ä¸ºåä¸ªç±»å«çIoU
        mIoU = np.nanmean(IoU)  # æ±åç±»å«IoUçå¹³å?
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # åFCNä¸­score.pyçfast_hist()å½æ°
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]  # label 取值0-1 predict 取值2-3!
        count = np.bincount(label, minlength=self.numClass ** 2)  # count every class has how much pixel
        confusionMatrix = count.reshape(self.numClass, self.numClass)

        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def DiceScore(self, imgPredict, imgLabel):
        imgLab = np.multiply(imgPredict, imgLabel)
        sumimgLab = np.sum(imgLab)
        sumimg = np.sum(imgPredict)
        sumLab = np.sum(imgLabel)
        # print( (2*sumimgLab), sumimg, sumLab )
        if (sumimg + sumLab == 0):
            discore = 0
        else:
            discore = (2 * sumimgLab) / (sumimg + sumLab)
        return discore

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def LoadData(r):
    dr = loadmat(r, verify_compressed_data_integrity=False)
    dd = dr['data']
    return dd


def GetBwData(data, thresh):
    datamin = np.min(data)
    datamax = np.max(data)
    data = (data - datamin) / (datamax - datamin)
    data[np.where(data < thresh)] = np.int32(0)
    data[np.where(data >= thresh)] = np.int32(1)
    return data


def GetNorData(data):
    datamin = np.min(data)
    datamax = np.max(data)
    data = (data - datamin) / (datamax - datamin)
    return data


def Get0BwData(data, thresh):
    d = data
    d[np.where(data >= thresh)] = 1
    d[np.where(data < thresh)] = 0
    return np.int32(d)


def GetArgMax(data):
    d = np.argmax(data, axis=0)  # max on channel c
    return d


def GetEvaluations(P, L, classnum, SP, SL):  # P阈值化预测值 SP正则化预测值,L=SL=label
    metric = SegmentationMetric(classnum)  # 3è¡¨ç¤ºæ?ä¸ªåç±»ï¼æå ä¸ªåç±»å°±å¡«å 
    metric.addBatch(P, L)
    pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy()
    cpa0, cpa1 = metric.classPixelAccuracy()

    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    discore = metric.DiceScore(SP, SL)
    BWdiscore = metric.DiceScore(P, L)

    # return pa, cpa, mpa, mIoU, FWIoU
    return pa, cpa0, cpa1, mpa, mIoU, FWIoU, discore, BWdiscore


def ReadXlsx(Fileroot, sheet_num):
    workbook = xlrd.open_workbook(Fileroot)
    sheet = workbook.sheet_by_index(sheet_num)
    return sheet


def GetMultiThresholdEva0(res, Labd, threshold):
    Pred = res[1, :, :]
    BwData = GetArgMax(res)  # 阈值化
    nordata = GetNorData(Pred.copy())  # 正则化
    Labd = np.int32(Labd.reshape(1, -1))
    BwData = np.int32(BwData.reshape(1, -1))
    nordata = (nordata.reshape(1, -1))
    pa, cpa0, cpa1, mpa, mIoU, FWIoU, discore, BWdiscore = GetEvaluations(BwData, Labd, 2, nordata, Labd)
    return [pa, cpa0, cpa1, mpa, mIoU, FWIoU, discore, BWdiscore]


def GetMultiThresholdEva(res, Labd, threshold):  # i think somthing wrong here
    Pred = res[1, :, :]
    nordata = GetNorData(Pred.copy())
    BwData = Get0BwData(nordata.copy(), threshold)
    Labd = np.int32(Labd.reshape(1, -1))
    BwData = np.int32(BwData.reshape(1, -1))
    nordata = (nordata.reshape(1, -1))
    pa, cpa0, cpa1, mpa, mIoU, FWIoU, discore, BWdiscore = GetEvaluations(BwData, Labd, 2, nordata, Labd)
    return [pa, cpa0, cpa1, mpa, mIoU, FWIoU, discore, BWdiscore]


def GetIouDice(res, labd):
    #cal class1's iou
    Pred = res[1, :, :]
    nordata = GetNorData(Pred.copy())
    inte = np.multiply(labd, nordata)
    intesum = np.sum(inte)
    Labdsum = np.sum(labd)
    nordsum = np.sum(nordata)
    iou = intesum / (Labdsum + nordsum - intesum)
    dice = 2 * intesum / (Labdsum + nordsum)
    return [iou, dice]


def CalIouDice(pred, labd):
    inte = np.multiply(labd, pred)
    intesum = np.sum(inte)
    Labdsum = np.sum(labd)
    nordsum = np.sum(pred)
    iou = intesum / (Labdsum + nordsum - intesum)
    dice = 2 * intesum / (Labdsum + nordsum)
    return iou, dice


def GetMeanIouDice(res, labd):
    Pred0 = res[0, :, :]
    Pred1 = res[1, :, :]
    labd0 = 1 - labd.copy()
    labd1 = labd.copy()
    nordata0 = GetNorData(Pred0.copy())
    nordata1 = GetNorData(Pred1.copy())
    iou0, dice0 = CalIouDice(nordata0.copy(), labd0.copy())
    iou1, dice1 = CalIouDice(nordata1.copy(), labd1.copy())
    iou = (iou0 + iou1) / 2.0
    dice = (dice0 + dice1) / 2.0
    return [iou, dice, iou0, dice0, iou1, dice1]


def CalIouDice2(pred, labd, thes):
    pred[pred < thes] = 0
    pred[pred > thes] = 1
    inte = np.multiply(labd, pred)

    intesum = np.sum(inte)
    Labdsum = np.sum(labd)
    nordsum = np.sum(pred)
    iou = intesum / (Labdsum + nordsum - intesum)
    dice = 2 * intesum / (Labdsum + nordsum)
    return iou, dice


def GetMeanIouDice2(res, labd, thes):
    Pred0 = res[0, :, :]
    Pred1 = res[1, :, :]
    labd0 = 1 - labd.copy()
    labd1 = labd.copy()
    nordata0 = GetNorData(Pred0.copy())
    nordata1 = GetNorData(Pred1.copy())
    iou0, dice0 = CalIouDice2(nordata0.copy(), labd0.copy(), thes)
    iou1, dice1 = CalIouDice2(nordata1.copy(), labd1.copy(), thes)
    iou = (iou0 + iou1) / 2.0
    dice = (dice0 + dice1) / 2.0
    return [iou, dice, iou0, dice0, iou1, dice1]