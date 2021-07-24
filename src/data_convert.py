import csv
import pandas as pd
import numpy

class DataConvert:

    ################
    #コンストラクタ
    ################
    def __init__(self):
        self.data_list = []

    ################################
    #価格データを読み込んで保持する。
    ################################
    def readData(self):
        with open("../data/USDJPY_M5_close.csv","r",encoding="utf8") as f:
            next(f)    
            reader = csv.reader(f)
            for row in reader:
                row.extend(["","","",""])
                self.data_list.append(row)

    #
    def setHighLowFlg(self):

        for i in range(len(self.data_list)):
            if i == 0:
                continue

            if i >= len(self.data_list) - 1:
                break

            if (float(self.data_list[i][2]) > float(self.data_list[i-1][2])  and
                float(self.data_list[i][2]) > float(self.data_list[i+1][2]) ):
                self.data_list[i][4] = 1
                #print(self.data_list[i])
            elif (float(self.data_list[i][2]) < float(self.data_list[i-1][2]) and
                  float(self.data_list[i][2]) < float(self.data_list[i+1][2]) ):
                self.data_list[i][4] = 0
                #print(self.data_list[i])


    def setLowestFlg(self):
        first = second = third = -1

        for i in range(len(self.data_list)):
            if self.data_list[i][4] == 0:
                if first < 0:
                    first = i
                    continue
                elif first >= 0 and second < 0:
                    second = i
                    continue
                elif first >= 0 and second >= 0 and third < 0:
                    third = i
                    continue
                if (self.data_list[second][2] < self.data_list[first][2] and
                    self.data_list[second][2] < self.data_list[third][2]):
                    self.data_list[second][5] = 0
                    self.data_list[second]
                    #print(self.data_list[second])

                first = second
                second = third
                third = i

    def setHighestFlg(self):
        first = second = third = -1

        for i in range(len(self.data_list)):
            if self.data_list[i][4] == 1:
                if first < 0:
                    first = i
                    continue
                elif first >= 0 and second < 0:
                    second = i
                    continue
                elif first >= 0 and second >= 0 and third < 0:
                    third = i
                    continue
                if (self.data_list[second][2] > self.data_list[first][2] and
                    self.data_list[second][2] > self.data_list[third][2]):
                    self.data_list[second][5] = 1
                    self.data_list[second]
                    #print(self.data_list[second])

                first = second
                second = third
                third = i

    #1,0の順に並ばない場合、1と1の間に0をセット、0と0の間に1をセット
    def dataFix(self):

        last_row = 0
        last = -1
        for i in range(len(self.data_list)):

            if self.data_list[i][5] == 0 or self.data_list[i][5] == 1:
                #last_row = i

                if self.data_list[i][5] == last:
                    self.putFlg2(last_row,i,self.data_list[i][5])
                    #self.data_list[i][6] = "BAD"
                last = self.data_list[i][5] 
                last_row = i

    #第２レベルの天底の欠損項目を追加する。
    def putFlg2(self,start,end,top_bottom):
        #0が連続した場合、1を追加する。
        if top_bottom == 0:
            wk_max = start
            for i in range(start+1,end):
                #print(self.data_list[wk_max][5])
                #print(self.data_list[i][5])
                
                if self.data_list[wk_max][2] < self.data_list[i][2]:
                    wk_max = i
            self.data_list[wk_max][5] = 1

        #1が連続した場合、0を追加する。
        elif top_bottom == 1:
            wk_min = start
            for i in range(start+1,end):
                if self.data_list[wk_min][2] > self.data_list[i][2]:
                    wk_min = i
            self.data_list[wk_min][5] = 0

    #天井度をセットする。
    def createCeilingIndex(self):

        start_i = 0
        end_i = 0


        #print(len(self.data_list))
        #quit()
        for i in range(len(self.data_list)):
            if self.data_list[i][5] == 1 or self.data_list[i][5] == 0:
                start_i = i
                end_i = self.getEndCeilingIndex(i)
                if end_i < 0:
                    continue

                #天井と底の始点インデックスと終点インデックスが取れたら
                #その間の天井度をセットする。
                self.setCeilingIndex(start_i,end_i)
            #print(i)


    #天井に対する底のインデックスまたは、底に対する天井のインデックスを取得する。
    def getEndCeilingIndex(self,start):
        #startが1なら次の0のインデックスを探す。
        #startが0なら次の1のインデックスを探す。
        #print("start=%d"% start)
        #quit()

        for i in range(len(self.data_list)):
            if i > len(self.data_list):
                break

            if self.data_list[start][5] == 1:
                for j in range(start + 1,len(self.data_list)):
                    if i > len(self.data_list):
                        break
                    if self.data_list[j][5] == 0:
                        return j

            elif self.data_list[start][5] == 0:
                for k in range(start + 1,len(self.data_list)):
                    if i > len(self.data_list):
                        break
                    if self.data_list[k][5] == 1:
                        return k

        return -1

    #startindexとendindexの間の指標を計算してセットする。
    def setCeilingIndex(self,start,end):
        #startの次がendの場合は何もしない。
        if end - start <= 1:
            return

        dist = 0
        if self.data_list[start][5] == 1:
            #print(self.data_list[start][2])
            #print(type(self.data_list[start][2]))
            
            dist = float(self.data_list[start][2]) - float(self.data_list[end][2])
            #print("start=%d"%start)
            #print("end=%d"%end)

            for i in range(start + 1,end):
                wk_dist = float(self.data_list[start][2]) - float(self.data_list[i][2])
                #print(float(self.data_list[start][2]))
                #print(float(self.data_list[i][2]))
                #print("dist=%f"%dist)
                #print("wk_dist=%f"%wk_dist)
                partial_dist = wk_dist / dist 
                #print("partial_dist=%f"%partial_dist)
                #quit()

                #self.data_list[i][5] = 1 - (dist - partial_dist)
                self.data_list[i][5] = 1 - partial_dist


        if self.data_list[start][5] == 0:
            dist = float(self.data_list[end][2]) - float(self.data_list[start][2])

            for i in range(start + 1,end):
                wk_dist = float(self.data_list[i][2]) - float(self.data_list[start][2])
                partial_dist = wk_dist / dist 
                self.data_list[i][5] = 0 + partial_dist


    #移動平均を出力する。
    def setAveragePrice(self,term):

        for i in range(len(self.data_list)):
            if i < term:
                continue

            total = 0.0

            for j in range(i - term,i):
                total += float(self.data_list[j][2])

            self.data_list[i][6] =  total / term


    #乖離 乖離率＝価格／移動平均線
    def setDeviation(self):

        for i in range(len(self.data_list)):

            if self.data_list[i][6] == '' or self.data_list[i][6] == None:
                continue

            #print(type(self.data_list[i][2]))
            #print(type(self.data_list[i][6]))
            #print(self.data_list[i][2])
            #print(self.data_list[i][6])

            self.data_list[i][7] = float(self.data_list[i][2]) / float(self.data_list[i][6])



    ###############
    #CSVを出力する。
    ###############
    def outputCsv(self):
        df = pd.DataFrame(self.data_list)
        df.to_csv('../data/USDJPY_M5_convert.csv', header=False, index=False)


    def run(self):
        self.readData()
        self.setHighLowFlg()
        self.setLowestFlg()
        self.setHighestFlg()
        self.dataFix()
        self.createCeilingIndex()
        self.setAveragePrice(60)
        self.setDeviation()
        self.outputCsv()


if __name__ == "__main__":

    DataConvert().run()

