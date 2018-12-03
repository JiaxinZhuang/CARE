from PIL import Image
import os
import numpy as np

def manhattanDistance(oPoint, iPoints):
    oPoint = np.array(oPoint)
    distance = min(np.sum(np.abs(iPoints - oPoint), axis=1))
    return distance

def converMask2DistanceMap(dataPath, outPath):
    for idx, filePath in enumerate(os.listdir(dataPath)):
        print(idx)
        path = os.path.join(dataPath, filePath)
        img=Image.open(path)
        #imgArr = np.array(img, dtype=np.float32)/255
        imgArrShape = imgArr.shape
        distanceMap = -1*np.ones(imgArrShape, dtype=np.int32)
        outPoints = np.argwhere(imgArr==1)
        inPoints = np.argwhere(imgArr==0)

        # get edge points
        left_top = inPoints[0]
        right_bottom = inPoints[-1]
        edgeInPoints = []
        for x,y in inPoints:
            if x > left_top[0] and x < right_bottom[0] and y > left_top[1] and y < right_bottom[1]:
                continue
            else:
                edgeInPoints.append([x,y])

        # get distance map for out Points
        for index, (x,y) in enumerate(outPoints):
            print('{}'.format(index/len(outPoints)))
            distance = manhattanDistance([x,y], edgeInPoints)
            distanceMap[x,y] = distance
        out = os.path.join(outPath, filePath)
        np.save(out, distanceMap)



if __name__ == '__main__':
    dataPath = "../data/mask/6_mask/"
    outPath = "../data/DistanceMap/6_distanceMap/"
    converMask2DistanceMap(dataPath, outPath)
