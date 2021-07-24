import numpy as np
import pandas as pd
import ActivationFunction as AF
from datetime import datetime as dt
from threelayernetwork import ThreeLayerNetwork


#教師データの作り方
#1~60のデータをinputする。
#feedworwadする。
#天井度が出力される。



class MainClass:

    def __init__(self):
        
        self.inodes = 60 #入力層を60個に設定
        self.hnodes = 60 #隠れ層をのニューロン数を60個に設定
        self.onodes = 1 #出力層を1個に設定
        self.lr = 0.3 #学習率を設定 0.01~0.5

        self.train_data = []
        self.test_data = []
        self.nn = '' #ニューラルネットワークオブジェクト

    #トレーニングデータの読み込み
    def readTrainData(self):
        self.train_data = pd.read_csv("../data/USDJPY_M5_convert_train.csv").values.tolist()


    #テストデータの読み込み
    def readTestData(self):
        self.test_data = pd.read_csv("../data/USDJPY_M5_convert_test.csv").values.tolist()






    #乖離率の配列を作成する。
    def getKairiList(self,start,end):
        ret = []
        for i in range(start,end):
            ret.append(self.train_data[i][7])
        return ret

    #テストデータから入力用の乖離率を取得する。
    def getKairiFromTest(self,start,end):
        idata = []
        for i in range(start,end):
            idata.append(self.test_data[i][7])          
        return idata

    #i番目のテストデータを取得する。
    def getDataFromTest(self,i):
        return self.test_data[i]


    #i番目のトレーニングデータを取得する。
    def getDataFromTrain(self,i):
        return self.train_data[i]

    #天井度の配列を作成する。
    def getCeilingList(self,start,end):
        ret = []
        for i in range(start,end):
            ret.append(self.train_data[i][5])
        return ret

    def getCeilingValue(self,num):
        return self.train_data[num][5]

    def getAnswer(self,end):
        return [self.test_data[end][0],self.test_data[end][1],self.test_data[end][5]]

    def getDateString(self):
        return dt.now().strftime('%Y%m%d%H%M%S')


    #教師データの作成
    def outputTeachData(self):

        #print("check predict")
        self.readTrainData()

        # テスト
        scoreboard = []
        output_data = []

        for i in range(len(self.train_data)):

            if i < self.inodes:
                continue

            idata = self.getKairiList(i-60, i)
            tdata = self.getDataFromTrain(i)

            predict = self.nn.feedForwardForTeachData(idata)
            output_data.append([tdata[0]+'.'+tdata[1],tdata[2],tdata[7],predict[0][0]])
            pass


        df = pd.DataFrame(output_data)
        df.to_csv("../data/teach_" + self.getDateString() + ".csv")


    ######################
    #教師データを使った学習
    ######################
    def learn(self):

        self.readTrainData()

        # 学習
        #epoch = 10
        epoch = 10

        for e in range(epoch):
            print('#epoch ', e)

            #データサイズを取得する。（mnistのデータは6万行）
            data_size = len(self.train_data)

            for i in range(data_size):
                #trainig_data_listの乖離率が設定されている行までスキップする。
                if self.train_data[i][5] == '' or self.train_data[i][5] == None:
                    continue

                #1000回ごとに現在の回数を表示
                if i % 1000 == 0:
                    print('  train: {0:>5d} / {1:>5d}'.format(i, data_size))

                #乖離率を求める。
                clist = self.getKairiList(i-60, i)

                #出力ノードの配列を初期化する。
                #tdata = np.zeros(self.onodes) + 0.5
                tdata = self.getCeilingValue(i)

                #誤差逆伝搬
                self.nn.backprop(clist, tdata)

                pass
            pass


    def checkPredict(self):

        #print("check predict")
        self.readTestData()

        # テスト
        scoreboard = []

        output_data = []

        for i in range(len(self.test_data)):

            if i < self.inodes:
                continue


            idata = self.getKairiFromTest(i-60,i)
            tdata = self.getDataFromTest(i)


            predict = self.nn.feedforward(idata)
            output_data.append([tdata[0]+'.'+tdata[1],tdata[2],predict[0][0]])
            pass


        df = pd.DataFrame(output_data)
        df.to_csv("../data/output_" + self.getDateString() + ".csv")

    def run(self):
        # ニューラルネットワークの初期化
        self.nn = ThreeLayerNetwork(self.inodes, self.hnodes, self.onodes, self.lr)

        self.learn()
        self.checkPredict()


    def runCreateTeachData(self):
        # ニューラルネットワークの初期化
        self.nn = ThreeLayerNetwork(self.inodes, self.hnodes, self.onodes, self.lr)

        self.outputTeachData()


if __name__=='__main__':

    #MainClass().run()
    MainClass().runCreateTeachData()






