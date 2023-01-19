# from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# import tensorflow as tf
from tensorflow import keras
import joblib

import pandas as pd
import numpy as np
# viz
# import hvplot.pandas
# import holoviews as hv
# import panel as pn
# def synchronize(dfx,dfy):
#     '''
#     synchronizes on index dfx and dfy and return tuple of synchronized data frames
#     Note: assumes dfy has only one column
#     '''
#     dfsync=pd.concat([dfx,dfy],axis=1).dropna()
#     return dfsync.iloc[:,:-1],dfsync.iloc[:,-1:]

def synchronize(dfx,dfy,lead_time=0,lead_freq='D'):
    '''
    synchronizes on index dfx and dfy and return tuple of synchronized data frames
    Note: assumes dfy has only one column
    '''
    if lead_time > 0:
        dfy.index = dfy.index.shift(-lead_time,freq=lead_freq)
    dfsync=pd.concat([dfx,dfy],axis=1).dropna()
    return dfsync.iloc[:,:-len(dfy.columns)],dfsync.iloc[:,-len(dfy.columns):]


def create_antecedent_inputs(df,ndays=8,window_size=11,nwindows=10):
    '''
    create data frame for CALSIM ANN input
    Each column of the input dataframe is appended by :-
    * input from each day going back to 7 days (current day + 7 days) = 8 new columns for each input
    * 11 day average input for 10 non-overlapping 11 day periods, starting from the 8th day = 10 new columns for each input

    Returns
    -------
    A dataframe with input columns = (8 daily shifted and 10 average shifted) for each input column

    '''
    arr1=[df.shift(n) for n in range(ndays)]
    dfr=df.rolling(str(window_size)+'D',min_periods=window_size).mean()
    arr2=[dfr.shift(periods=(window_size*n+ndays),freq='D') for n in range(nwindows)]
    df_x=pd.concat(arr1+arr2,axis=1).dropna()# nsamples, nfeatures
    return df_x

def trim_output_to_index(df,index):
    '''
    helper method to create output of a certain size ( typically to match the input )
    '''
    return df.loc[index,:] #nsamples, noutput

def split(df, calib_slice, valid_slice):
    if type(calib_slice) == list:
        calib_set = pd.concat([df[slc] for slc in calib_slice],axis=0)
    else:
        calib_set = df[calib_slice]
    if type(valid_slice) == list:
        valid_set = pd.concat([df[slc] for slc in valid_slice],axis=0)
    else:
        valid_set = df[valid_slice]
    return calib_set, valid_set

class myscaler():
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = -float('inf')
    
    def fit_transform(self, data):
        data[data==-2]=float('nan')
        self.min_val = data.min()
        self.max_val = data.max()
        
    def update(self, other_scaler):
        self.min_val = np.minimum(self.min_val,other_scaler.min_val)
        self.max_val = np.maximum(self.max_val,other_scaler.max_val)
    
        
    def transform(self, data):
        return (data - self.min_val) * 1.0 / (self.max_val - self.min_val)
    
    def inverse_transform(self, data):
        if type(data)==np.ndarray:
            max_val = self.max_val.to_numpy().reshape(1,-1)
            min_val = self.min_val.to_numpy().reshape(1,-1)
            return data * (max_val - min_val) + min_val
        else:
            return data * (self.max_val - self.min_val) + self.min_val
    
def create_xyscaler(dfin,dfout):
    # xscaler=MinMaxScaler()
    xscaler=myscaler()
    _ = xscaler.fit_transform(pd.concat(dfin,axis=0))
    #
    yscaler=myscaler()
    _ = yscaler.fit_transform(pd.concat(dfout,axis=0))
    return xscaler, yscaler

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def dropout(x, p=0.03):
    return x * (np.random.binomial([np.ones(x.shape)],
                                  1-p)[0]) # * (1.0/(1-p)) # re-normalize the vector to compensate for 0's


def apply_augmentation(x, apply_aug=False,noise_sigma=0.03,dropout_ratio=0):
    np.random.seed(0)
    if apply_aug:
        x = jitter(x,sigma=noise_sigma)
        x = dropout(x,p=dropout_ratio)
    return x

def create_training_sets(dfin, dfout, calib_slice=slice('1940','2015'), valid_slice=slice('1923','1939'),
                         train_frac=None,
                         ndays=8,window_size=11,nwindows=10,
                         noise_sigma=0.,dropout_ratio=0.,
                         lead_time=0,lead_freq='D',
                         xscaler=None, yscaler=None):
    '''
    dfin is a dataframe that has sample (rows/timesteps) x nfeatures 
    dfout is a dataframe that has sample (rows/timesteps) x 1 label
    Both these data frames are assumed to be indexed by time with daily timestep

    This calls create_antecedent_inputs to create the CALSIM 3 way of creating antecedent information for each of the features

    Returns a tuple of two pairs (tuples) of calibration and validation training set where each set consists of input and output
    it also returns the xscaler and yscaler in addition to the two tuples above
    '''
    # create antecedent inputs aligned with outputs for each pair of dfin and dfout
    dfina,dfouta=[],[]
    # scale across all inputs and outputs
    if (xscaler is None) or (yscaler is None):
        xscaler,yscaler=create_xyscaler(dfin,dfout)
        
    for dfi,dfo in zip(dfin,dfout):
        dfi,dfo=synchronize(dfi,dfo,lead_time=lead_time,lead_freq=lead_freq)
        dfi,dfo=pd.DataFrame(xscaler.transform(dfi),dfi.index,columns=dfi.columns),pd.DataFrame(yscaler.transform(dfo),dfo.index,columns=dfo.columns)
        dfi,dfo=synchronize(create_antecedent_inputs(dfi,ndays=ndays,window_size=window_size,nwindows=nwindows),dfo)
        dfina.append(dfi)
        dfouta.append(dfo)
    # split in calibration and validation slices
    if train_frac is None:
        dfins=[split(dfx,calib_slice,valid_slice) for dfx in dfina]
        dfouts=[split(dfy,calib_slice,valid_slice) for dfy in dfouta]
    else:
        train_sample_index = dfina[0].sample(frac=train_frac,random_state=0).index
        dfins=[(dfx.loc[dfx.index.isin(train_sample_index)],dfx.loc[~dfx.index.isin(train_sample_index)]) for dfx in dfina]
        dfouts=[(dfy.loc[dfy.index.isin(train_sample_index)],dfy.loc[~dfy.index.isin(train_sample_index)]) for dfy in dfouta]
        print('Randomly selecting %d samples for training, %d for test' % (dfins[0][0].shape[0],dfins[0][1].shape[0]))

    # append all calibration and validation slices across all input/output sets
    xallc,xallv=dfins[0]
    for xc,xv in dfins[1:]:
        xallc=np.append(apply_augmentation(xallc,noise_sigma=noise_sigma,dropout_ratio=dropout_ratio),xc,axis=0)
        xallv=np.append(xallv,xv,axis=0)
    yallc, yallv = dfouts[0]
    for yc,yv in dfouts[1:]:
        yallc=np.append(yallc,yc,axis=0)
        yallv=np.append(yallv,yv,axis=0)
    return (xallc,yallc),(xallv,yallv),xscaler,yscaler

def create_memory_sequence_set(xx,yy=None,time_memory=120):
    '''
    given an np.array of xx (features/inputs) and yy (labels/outputs) and a time memory of steps
    shape[0] of the array represents the steps (usually evenly spaced time)
    return a tuple of inputs/outputs sampled for every step going back to time memory
    The shape of the returned arrays is dictated by keras 
    inputs.shape (nsamples x time_memory steps x nfeatures)
    outputs.shape (nsamples x nlabels)
    '''
    xxarr=[xx.iloc[i:time_memory+i,:] for i in range(xx.shape[0]-time_memory)]
    xxarr=np.expand_dims(xxarr,axis=0)[0]
    if yy is not None:
        yyarr=[yy.iloc[time_memory+i,:] for i in range(xx.shape[0]-time_memory)]
        yyarr=np.array(yyarr)
        return xxarr,yyarr
    else:
        return xxarr
############### TESTING - SPLIT HERE #####################

def predict(model,dfx,xscaler,yscaler,columns=['prediction'],ndays=8,window_size=11,nwindows=10,verbose=0):
    dfx=pd.DataFrame(xscaler.transform(dfx),dfx.index,columns=dfx.columns)
    xx=create_antecedent_inputs(dfx,ndays=ndays,window_size=window_size,nwindows=nwindows)
    oindex=xx.index
    yyp=model.predict(xx, verbose=verbose)
    dfp=pd.DataFrame(yscaler.inverse_transform(yyp),index=oindex,columns=columns)
    return dfp

def predict_with_actual(model, dfx, dfy, xscaler, yscaler):
    dfp=predict(model, dfx, xscaler, yscaler)
    return pd.concat([dfy,dfp],axis=1).dropna()
    
def plot(dfy,dfp):
    return dfy.hvplot(label='target')*dfp.hvplot(label='prediction')

# def show_performance(model, dfx, dfy, xscaler, yscaler):
#     dfyp=predict_with_actual(model,dfx,dfy,xscaler,yscaler)
#     print('R^2 ',r2_score(dfyp.iloc[:,0],dfyp.iloc[:,1]))
#     dfyp.columns=['target','prediction']
#     plt=(dfyp.iloc[:,1]-dfyp.iloc[:,0]).hvplot.kde().opts(width=300)+dfyp.hvplot.points(x='target',y='prediction').opts(width=300)
#     return pn.Column(plt, plot(dfyp.iloc[:,0],dfyp.iloc[:,1]))
###########
class ANNModel:
    '''
    model consists of the model file + the scaling of inputs and outputs
    '''
    def __init__(self,model,xscaler,yscaler):
        self.model=model
        self.xscaler=xscaler
        self.yscaler=yscaler
    def predict(self, dfin,columns=['prediction'],ndays=8,window_size=11,nwindows=10):
        return predict(self.model,dfin,self.xscaler,self.yscaler,columns=columns,ndays=ndays,window_size=window_size,nwindows=nwindows)
#
def save_model(location, model, xscaler, yscaler):
    '''
    save keras model and scaling to files
    '''
    joblib.dump((xscaler,yscaler),'%s_xyscaler.dump'%location)
    model.save('%s.h5'%location)

def load_model(location,custom_objects):
    '''
    load model (ANNModel) which consists of model (Keras) and scalers loaded from two files
    '''
    model=keras.models.load_model('%s.h5'%location,custom_objects=custom_objects)
    xscaler,yscaler=joblib.load('%s_xyscaler.dump'%location)
    return ANNModel(model,xscaler,yscaler)

########### TRAINING - SPLIT THIS MODULE HERE ###################

def train_nn(x,y,hidden_layer_sizes=(10,),max_iter=1000,activation='relu',tol=1e-4):
    mlp=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iter,activation=activation, tol=tol)
    mlp.fit(x,y)
    return mlp

def _old_predict(df_x, mlp, xs, ys):
    y_pred=mlp.predict(xs.transform(df_x))
    y_pred=ys.inverse_transform(np.vstack(y_pred))
    return pd.DataFrame(y_pred,df_x.index,columns=['prediction'])

# def show(df_x, df_y, mlp, xs, ys):
#     y=np.ravel(ys.transform(df_y))
#     y_pred=mlp.predict(xs.transform(df_x))
#     r2=mlp.score(xs.transform(df_x),y)
#     print('Score: ',r2)
#     return pn.Column(pn.Row(hv.Scatter((y,y_pred)).opts(aspect='square'),hv.Distribution(y_pred-y).opts(aspect='square')),
#                      hv.Curve((df_y.index,y_pred),label='prediction')*hv.Curve((df_y.index,y),label='target').opts(width=800))

def train(df_x,df_y,hidden_layer_sizes=(10,),max_iter=1000,activation='relu',tol=1e-4):
    xs=MinMaxScaler()
    x=xs.fit_transform(df_x)
    ys=MinMaxScaler()
    y=np.ravel(ys.fit_transform(df_y))
    mlp=train_nn(x, y,hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iter,activation=activation,tol=tol)
    return mlp, xs, ys

def train_more(mlp,xs,ys,df_x,df_y):
    x=xs.transform(df_x)
    y=np.ravel(ys.transform(df_y))
    mlp.fit(x,y)
    return mlp

parameters = {
    "kernel_initializer": "he_normal"
}

def basic_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
):
    """
    A one-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    # axis = -1 if keras.backend.image_data_format() == "channels_last" else 1


    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.ZeroPadding1D(padding=1,name="padding{}{}_branch2a".format(stage_char, block_char))(x)
        y = keras.layers.Conv1D(filters,kernel_size,strides=stride,use_bias=False,
                                name="res{}{}_branch2a".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding1D(padding=1,name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv1D(filters,kernel_size,use_bias=False,
                                name="res{}{}_branch2b".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)

        if block == 0:
            shortcut = keras.layers.Conv1D(filters,1,strides=stride,use_bias=False,
                                           name="res{}{}_branch1".format(stage_char, block_char),
                                           **parameters)(x)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        
        y = keras.layers.Activation("relu",name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
):
    """
    A one-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    """
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2

    # axis = -1 if keras.backend.image_data_format() == "channels_last" else 1


    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv1D(filters,1,strides=stride,use_bias=False,
                                name="res{}{}_branch2a".format(stage_char, block_char),
                                **parameters)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu",name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding1D(padding=1,name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv1D(filters,kernel_size,use_bias=False,
                                name="res{}{}_branch2b".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu",name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv1D(filters * 4, 1, use_bias=False,
                                name="res{}{}_branch2c".format(stage_char, block_char),
                                **parameters)(y)
        y = keras.layers.BatchNormalization()(y)

        if block == 0:
            shortcut = keras.layers.Conv1D(filters * 4, 1, strides=stride, use_bias=False,
                                           name="res{}{}_branch1".format(stage_char, block_char),
                                           **parameters)(x)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu",name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

def conv_filter_generator(ndays=7,window_size = 11, nwindows=10):
    w = np.zeros((1,ndays+nwindows*window_size,ndays+nwindows))
    for ii in range(ndays):
        w[0,ndays+nwindows*window_size-ii-1,ii] = 1
    for ii in range(nwindows):
        w[0,((nwindows-ii-1)*window_size):((nwindows-ii)*window_size),ndays+ii] = 1/window_size
    return w
