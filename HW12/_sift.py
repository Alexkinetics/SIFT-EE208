import cv2
import numpy as np
from math import sin, cos, pi
import random


def _Gray(img):
    x, y, z = np.shape(img)
    gray = np.zeros([x,y], "uint8")# 2 unsigned int 8
    for i in range(x):
        for j in range(y):
            gray[i][j] = np.dot(np.array(img[i][j],
                dtype="float"), [.114, .587, .299])
    return gray


def _Sift(img):

    r, c = np.shape(img)
    corners = [[int(i[0][0]), int(i[0][1])]
               for i in cv2.goodFeaturesToTrack(img, 233, 0.01, 10)]
    img = cv2.GaussianBlur(img, (5, 5), 1, 1)
    img = np.array(img, dtype="float")

    def _Grad(img):
        x, y = r, c

        # sobel效果不好
        # s_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float")  # X方向
        # s_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="float")

        kernel = np.array([
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]], dtype="float") / 6
        gx = cv2.filter2D(img, -1, np.array(kernel[1]))
        gy = cv2.filter2D(img, -1, np.array(kernel[0]))
        gradient = np.zeros([x, y], "float")
        angle = np.zeros([x, y], "float")
        for i in range(x):
            for j in range(y):
                gradient[i][j] = ((gx[i][j]) ** 2 + (gy[i][j]) ** 2) ** 0.5
                angle[i][j] = np.math.atan2(gy[i][j], gx[i][j])
        return gradient, angle

    gradient, angle = _Grad(img)
    bins = (r + c) // 80 #to vote

    length = len(corners)

    def _Vote():
        direct = [] # 存储每个角点的主方向
        for corner in corners:
            y, x = corner
            voting = [0 for i in range(37)]
            for i in range(max(x-bins,0), min(x+bins+1,r)):
                for j in range(max(y-bins,0), min(y+bins+1,c)):
                    k = int((angle[i][j]+pi) / (pi/18) + 1)
                    if k >= 37:
                        k = 36
                    voting[k] += gradient[i][j]
            # find max
            p=1
            for i in range(2,37):
                if voting[i]>voting[p]:
                    p=i

            direct.append((p/18 - 1- 1/36) * pi)
        return direct

    direct = _Vote()


    def _Feature(pos, theta):
        def _theta(x, y):
            if (x < 0 or x >= r) or (y < 0 or y >= c):
                return 0
            dif = angle[x][y] - theta
            return dif if dif > 0 else dif + 2 * pi

        def _DB_linear(x, y):
            xx, yy = int(x), int(y)
            dy1, dy2 = y-yy, yy+1-y
            dx1, dx2 = x-xx, xx+1-x
            val = _theta(xx,yy)*dx2*dy2 \
                  + _theta(xx+1,yy)*dx1*dy2 \
                  + _theta(xx,yy+1)*dx2*dy1 \
                  + _theta(xx+1,yy+1)*dx1*dy1
            return val

        y0, x0 = pos
        H = np.array([cos(theta), sin(theta)])
        V = np.array([-sin(theta),cos(theta)])

        val = []
        def cnt(x1, x2, y1, y2, xsign, ysign):
            voting = [0 for i in range(9)]
            for x in range(x1, x2):
                for y in range(y1, y2):
                    dp = [x * xsign, y * ysign]
                    p = H * dp[0] + V * dp[1]
                    bin = int((_DB_linear(p[0]+x0, p[1]+y0))//(pi/4) + 1)
                    if bin > 8:
                        bin = 8
                    voting[bin] += 1
            return voting[1:]

        bins = (r + c) // 150
        for xsign in [-1,1]:
            for ysign in [-1,1]:
                val += cnt(0, bins, 0, bins, xsign, ysign)
                val += cnt(bins, bins*2, 0, bins, xsign, ysign)
                val += cnt(bins, bins*2, bins, bins*2, xsign, ysign)
                val += cnt(0, bins, bins, bins*2, xsign, ysign)
        return val


    feature = []
    for i in range(length):
        val = _Feature(corners[i], direct[i])
        m = sum(k * k for k in val) ** 0.5
        l = [k / m for k in val]
        feature.append(l)
    return feature, corners, length




def _Merge(img1, img2):
    h1, w1 ,a= np.shape(img1)
    h2, w2 ,a= np.shape(img2)
    if h1 < h2:
        extra = np.array([[[0,0,0] for i in range(w1)] for ii in range(h2-h1)])
        img1 = np.vstack([img1, extra])
    elif h1 > h2:
        extra = np.array([[[0,0,0] for i in range(w2)] for ii in range(h1-h2)])
        img2 = np.vstack([img2, extra])
    return np.hstack([img1,img2])


def _Match(threshold):
    for id in range(len(imgset)):
        x = []
        cnt = 0
        for i in range(lt):
            tmp = []
            for j in range(ll[id]):
                sc= np.inner(np.array(ft[i]), np.array(ff[id][j]))
                tmp.append(sc)
            x.append([tmp.index(max(tmp)), max(tmp)])
        for a in range(len(x)):
            b, s = x[a]
            if s < threshold:
                continue
            cnt += 1
            color = ((random.randint(0, 255)),
                     (random.randint(0, 255)),
                     (random.randint(0, 255)))
            cv2.line(mgimgs[id], tuple(ct[a]),
                     tuple([cc[id][b][0] + w,
                            cc[id][b][1]]), color, 1)
        if cnt > 6:
            cv2.imwrite("match%d.jpg" % id, mgimgs[id])
            print("MATCHED %d" % id)
            img = np.array(mgimgs[id], dtype="uint8")
            cv2.namedWindow("MATCH_RESULT")
            cv2.imshow("MATCH_RESULT", img)
            cv2.waitKey(0)
            cv2.destroyWindow("MATCH_RESULT")

        else:
            print("NOT %d" % id)

    return


if __name__ == "__main__":

    ### SIFT ###
    tgt0 = cv2.imread(r"target.jpg", 1)
    imgset0 = [cv2.imread("%d.jpg" % i, 1) for i in range(1, 6)]
    r0,c0,a0=np.shape(tgt0)
    times=1.0
    resized_tgt0=cv2.resize(tgt0,(int(r0*times),int(c0*times)))
    # 灰度化
    tgt = _Gray(resized_tgt0)
    imgset = [_Gray(imgset0[i]) for i in range(len(imgset0))]

    ff = []
    cc = []
    ll = []
    ft, ct, lt = _Sift(tgt)
    for i in range(len(imgset)):
        f, c, l = _Sift(imgset[i])
        ff.append(f)
        cc.append(c)
        ll.append(l)

    w = np.shape(tgt)[1]
    mgimgs = [_Merge(tgt0, imgset0[i]) for i in range(len(imgset0))]
    print("All Original Pics Processed!")

    _Match(0.8)




