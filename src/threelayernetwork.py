import numpy as np
import pandas as pd
import ActivationFunction as AF
from datetime import datetime as dt


# 3層ニューラルネットワーク
class ThreeLayerNetwork:

    ######################
    # コンストラクタ
    ######################
    def __init__(self, inodes, hnodes, onodes, lr):

        # 各レイヤーのノード数
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        # 学習率
        self.lr = lr

        # 重みの初期化
        #正規分布に従った乱数の発生
        self.w_ih = np.random.normal(0.0, 1.0, (self.hnodes, self.inodes))
        self.w_ho = np.random.normal(0.0, 1.0, (self.onodes, self.hnodes))

        self.tw_ih = np.zeros((self.hnodes, self.inodes)) + 0.1
        self.tw_ho = np.zeros((self.hnodes, self.inodes)) + 0.1

        #print(self.tw_ih)
        #print(self.tw_ho)
        #quit()

        # 活性化関数
        self.af = AF.sigmoid
        self.daf = AF.derivative_sigmoid
        self.relu = AF.relu

    #############
    # 誤差逆伝搬
    #############
    def backprop(self, idata, tdata):

        # 縦ベクトルに変換 #型をそろえて配列に格納する。
        o_i = np.array(idata, ndmin=2).T
        t = np.array(tdata, ndmin=2).T

        # 隠れ層 #ベクトルのドット積を計算する。
        x_h = np.dot(self.w_ih, o_i)
        o_h = self.af(x_h)
        
        # 出力層
        x_o = np.dot(self.w_ho, o_h)
        o_o = self.af(x_o)

        # 誤差計算
        e_o = (t - o_o)
        e_h = np.dot(self.w_ho.T, e_o)

        # 重みの更新
        self.w_ho += self.lr * np.dot((e_o * self.daf(o_o)), o_h.T)
        self.w_ih += self.lr * np.dot((e_h * self.daf(o_h)), o_i.T)


    #########
    # 順伝搬
    #########
    def feedforward(self, idata):
        # 入力のリストを縦ベクトルに変換
        o_i = np.array(idata, ndmin=2).T

        # 隠れ層
        x_h = np.dot(self.w_ih, o_i)
        o_h = self.af(x_h)

        # 出力層
        x_o = np.dot(self.w_ho, o_h)
        o_o = self.af(x_o)

        return o_o

    #
    def feedForwardForTeachData(self, idata):
        # 入力のリストを縦ベクトルに変換
        o_i = np.array(idata, ndmin=2).T

        # 隠れ層
        x_h = np.dot(self.tw_ih, o_i)
        o_h = self.af(x_h)

        # 出力層
        x_o = np.dot(self.tw_ho, o_h)
        o_o = self.af(x_o)

        return o_o