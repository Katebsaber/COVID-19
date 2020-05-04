import cv2
from PIL import Image
from glob import glob
import pandas as pd
import numpy as np


def imResize(impPath: str, outPath: str, shape: tuple, save: bool):
    """ Note that imResize would corrupt the ratio"""
    img = cv2.imread(impPath)
    res = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
    if save:
        Image.fromarray(res).save(outPath)
    return res


def preprocess():
    # Preprocess
    # size = 512, 512
    #
    # PFiles = glob('data/CT_COVID/*.png')
    # NFilesPNG = glob('data/CT_NonCOVID/*.png')
    # NFilesJPG = glob('data/CT_NonCOVID/*.jpg')

    # for pSample in PFiles:
    #     fileName = pSample[14:-4]
    #     res = imResize(impPath=pSample, outPath=f"data/preprocessed/P/{fileName}.jpg", shape=size, save=True)

    # for nSample in NFilesJPG:
    #     fileName = nSample[17:-4]
    #     res = imResize(impPath=nSample, outPath=f"data/preprocessed/N/{fileName}.jpg", shape=size, save=True)

    # for nSample in NFilesPNG:
    #     fileName = nSample[17:-4]
    #     res = imResize(impPath=nSample, outPath=f"data/preprocessed/N/{fileName}.jpg", shape=size, save=True)
    pass


def sift(impPath: str, nfeatures: int, outShape: tuple):
    img = cv2.imread(impPath)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    kp, desc = sift.detectAndCompute(img, None)

    desc = desc.reshape(outShape)
    return desc


if __name__ == '__main__':
    nFeatures = 100

    # SIFT
    files = []
    PFiles = glob('data/preprocessed/P/*.jpg')
    for p in PFiles:
        files.append((p, 1))
    NFiles = glob('data/preprocessed/N/*.jpg')
    for n in NFiles:
        files.append((n, -1))

    SiftFeatures = []
    fileName = []
    label = []
    for sample, l in files:
        d = sift(impPath=sample, nfeatures=nFeatures, outShape=(1, -1))
        if d.shape == (1, 128 * nFeatures):
            SiftFeatures.append(d)
            fileName.append(sample.split("/")[-1])
            label.append(l)

    df = pd.DataFrame(np.concatenate(SiftFeatures))
    df['fileName'] = fileName
    df['label'] = label

    print(df.head())
    print(df.tail())

    df.to_csv(f"output/data_{nFeatures}.csv")
    df.to_excel(f"output/data_{nFeatures}.xlsx")
