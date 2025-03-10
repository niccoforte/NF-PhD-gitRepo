import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import sklearn

### General Data Initialization


def load_TrainTestData(CSV_train_in, CSV_train_out, CSV_val_in, CSV_val_out, CSV_test_in, CSV_test_out):
    train_in = pd.read_csv(CSV_train_in, index_col=0, header=0).to_numpy()
    train_out = pd.read_csv(CSV_train_out, index_col=0, header=0).to_numpy()
    val_in = pd.read_csv(CSV_val_in, index_col=0, header=0).to_numpy()
    val_out = pd.read_csv(CSV_val_out, index_col=0, header=0).to_numpy()
    test_in = pd.read_csv(CSV_test_in, index_col=0, header=0).to_numpy()
    test_out = pd.read_csv(CSV_test_out, index_col=0, header=0).to_numpy()
    return train_in, train_out, val_in, val_out, test_in, test_out

def load_perData(INcsv, OUTcsv):
    IN_df = pd.read_csv(INcsv, index_col=0).sort_index()
    OUT_df = pd.read_csv(OUTcsv, index_col=0).sort_index()
    perIN_df = IN_df.loc[:0]
    perOUT_df = OUT_df.loc[:0]
    return perIN_df.to_numpy()[0], perOUT_df.to_numpy()[:,1:]


def dataParams(x):
    return [np.min(x), np.max(x), np.mean(x), np.std(x)]

def standardize(x, minx, maxx, mode=0):
    if mode == 0:
        return (x - minx)/(maxx - minx)
    if mode == 1:
        return (x*(maxx - minx)) + minx
    
def normalize(x, mean, std, mode=0):
    if mode == 0:
        return (x - mean)/std
    if mode == 1:
        return (x*std) + mean

class Dataset_(Dataset):
    def __init__(self, x, y, edge_index=None):
        self.x = x
        self.y = y
        self.edge_index=edge_index
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]
    

class DATA:
    def __init__(self, path=0, load=False, LAT="FCC", dis="dN", dN=20, model="MLP", format=0):
        self.path = path
        self.LAT = LAT
        self.dis = dis
        self.dN = dN
        self.model = model

        self.get_DataPath()

        if load:
            self.get_DataFiles()
            self.load_data()

            if format == 1 and model.lower() == "mlp":
                self.load_DisDist_v1()
            elif format == 2 and model.lower() == "mlp":
                self.load_DisDist_v2()


    def get_DataPath(self):
        pData = 'data/'

        pAl          = pData + 'Al/'
        pAK          = pAl + 'AK/'
        pUTdisNodes  = pAK + 'Ductile-disNodes-FCC-12X16/'
        pUTdisNodes2 = pAK + '20_RD02_10mm/'
        pUTdisStruts = pAK + 'Ductile-disStruts-FCC-12X16/'
        pFTdisNodes  = pAK + 'Fracture-disNodes/'

        pTi    = pData + 'Ti/'
        pTiLAT = pTi + f'{self.dN}dN/{self.LAT}/'

        if self.path == 0:
            self.PATH  = pUTdisNodes2
        elif self.path == 1:
            self.PATH  = pTiLAT

    def get_DataFiles(self):
        self.CSV_train_in  = self.PATH + f'NN-UT-{self.dis}-trainIN.csv'
        self.CSV_train_out = self.PATH + f'NN-UT-{self.dis}-trainOUT.csv'
        self.CSV_val_in  = self.PATH + f'NN-UT-{self.dis}-valIN.csv'
        self.CSV_val_out = self.PATH + f'NN-UT-{self.dis}-valOUT.csv'
        self.CSV_test_in  = self.PATH + f'NN-UT-{self.dis}-testIN.csv'
        self.CSV_test_out = self.PATH + f'NN-UT-{self.dis}-testOUT.csv'

        self.INcsv = self.PATH + f'Ductile-disNodes-IN.csv'
        self.OUTcsv = self.PATH + f'Ductile-disNodes-OUT.csv'

    def load_data(self):
        train_in, train_out, val_in, val_out, test_in, test_out = load_TrainTestData(self.CSV_train_in, 
                                                                                     self.CSV_train_out, 
                                                                                     self.CSV_val_in, 
                                                                                     self.CSV_val_out, 
                                                                                     self.CSV_test_in, 
                                                                                     self.CSV_test_out)
        if self.model.lower() == "mlp":
            self.train_in, self.train_out = train_in, train_out
            self.val_in, self.val_out = val_in, val_out
            self.test_in, self.test_out = test_in, test_out
        elif self.model.lower() == "gnn":
            self.train_in = train_in.reshape(train_in.shape[:-1], train_in.shape[-1]//2, 2)
            self.train_out = train_out.reshape(train_out.shape[:-1], train_out.shape[-1]//2, 2)
            self.val_in = val_in.reshape(val_in.shape[:-1], val_in.shape[-1]//2, 2)
            self.val_out = val_out.reshape(val_out.shape[:-1], val_out.shape[-1]//2, 2)
            self.test_in = test_in.reshape(test_in.shape[:-1], test_in.shape[-1]//2, 2)
            self.test_out = test_out.reshape(test_out.shape[:-1], test_out.shape[-1]//2, 2)
        self.perIN, self.perOUT = load_perData(self.INcsv, self.OUTcsv)

        self.inParams = dataParams(np.concatenate((self.train_in, self.val_in, self.test_in)))
        self.outParams = dataParams(np.concatenate((self.train_out, self.val_out, self.test_out)))

        self.train_inST = standardize(self.train_in, self.inParams[0], self.inParams[1])
        self.train_outST = standardize(self.train_out, self.outParams[0], self.outParams[1])
        self.val_inST = standardize(self.val_in, self.inParams[0], self.inParams[1])
        self.val_outST = standardize(self.val_out, self.outParams[0], self.outParams[1])
        self.test_inST = standardize(self.test_in, self.inParams[0], self.inParams[1])
        self.test_outST = standardize(self.test_out, self.outParams[0], self.outParams[1])
        
        self.train_inNM = normalize(self.train_in, self.inParams[2], self.inParams[3])
        self.train_outNM = normalize(self.train_out, self.outParams[2], self.outParams[3])
        self.val_inNM = normalize(self.val_in, self.inParams[2], self.inParams[3])
        self.val_outNM = normalize(self.val_out, self.outParams[2], self.outParams[3])
        self.test_inNM = normalize(self.test_in, self.inParams[2], self.inParams[3])
        self.test_outNM = normalize(self.test_out, self.outParams[2], self.outParams[3])
    
    def load_DisDist_v1(self):
        train_in1 = self.perIN.reshape(len(self.perIN)//2, 2)
        self.train_in1 = np.array([i for i in train_in1 if max(train_in1[:,0]) != i[0] and min(train_in1[:,0]) != i[0] and 
                                                           max(train_in1[:,1]) != i[1] and min(train_in1[:,1]) != i[1]])

        train_out1 = self.train_in.reshape(len(self.train_in),len(self.train_in[0])//2,2)
        self.dx_out1 = train_out1[:,:,0].reshape(len(self.train_in),len(self.train_in[0])//2,1)
        self.dy_out1 = train_out1[:,:,1].reshape(len(self.train_in),len(self.train_in[0])//2,1)

        self.inParams1 = dataParams(self.train_in1)
        self.train_in1ST = standardize(self.train_in1, self.inParams1[0], self.inParams1[1])
        self.train_in1NM = normalize(self.train_in1, self.inParams1[2], self.inParams1[3])

        self.outParams1dx = dataParams(self.dx_out1)
        self.outParams1dy = dataParams(self.dy_out1)
        self.dx_out1ST = standardize(self.dx_out1, self.outParams1dx[0], self.outParams1dx[1])
        self.dy_out1ST = standardize(self.dy_out1, self.outParams1dy[0], self.outParams1dy[1])
        self.dx_out1NM = normalize(self.dx_out1, self.outParams1dx[2], self.outParams1dx[3])
        self.dy_out1NM = normalize(self.dy_out1, self.outParams1dy[2], self.outParams1dy[3])
    
    def load_DisDist_v2(self):
        train_in1 = self.perIN.reshape(len(self.perIN)//2, 2)
        self.train_in1 = np.array([i for i in train_in1 if max(train_in1[:,0]) != i[0] and min(train_in1[:,0]) != i[0] and 
                                                           max(train_in1[:,1]) != i[1] and min(train_in1[:,1]) != i[1]])
        self.train_in2 = np.array([self.train_in1.flatten()]*2)

        train_out2 = self.train_in.reshape(len(self.train_in),len(self.train_in[0])//2,2)
        dx_out2 = train_out2[:,:,0].reshape(len(self.train_in),len(self.train_in[0])//2)
        self.dx_out2 = np.stack((dx_out2, dx_out2), axis=1)
        dy_out2 = train_out2[:,:,1].reshape(len(self.train_in),len(self.train_in[0])//2)
        self.dy_out2 = np.stack((dy_out2, dy_out2), axis=1)

        self.inParams2 = dataParams(self.train_in2)
        self.train_in2ST = standardize(self.train_in2, self.inParams2[0], self.inParams2[1])
        self.train_in2NM = normalize(self.train_in2, self.inParams2[2], self.inParams2[3])

        self.outParams2dx = dataParams(self.dx_out2)
        self.outParams2dy = dataParams(self.dx_out2)
        self.dx_out2ST = standardize(self.dx_out2, self.outParams2dx[0], self.outParams2dx[1])
        self.dx_out2ST = standardize(self.dx_out2, self.outParams2dy[0], self.outParams2dy[1])
        self.dx_out2NM = normalize(self.dx_out2, self.outParams2dx[2], self.outParams2dx[3])
        self.dx_out2NM = normalize(self.dx_out2, self.outParams2dy[2], self.outParams2dy[3])

