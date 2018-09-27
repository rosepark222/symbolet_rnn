import csv
import re
import numpy as np

#    https://keras.io/getting-started/sequential-model-guide/#examples
#  check this out
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence 
from tensorflow.python.keras.models import load_model


batch_size = 500
epoch_size = 1000
#symbol_list = ['0_1_1' '2_1_1' '7_1_1' 'c_1_1']  ==> acc=0.95 by 60 epoch

xsymbol_list = ['0_1_1', '1_1_1', '2_1_1', '3_1_1',  '4_1_1', '4_2_nose',  
               '5_1_1',  '5_2_hook', '6_1_1',  '7_1_1', '8_1_1', '9_1_1', 
               'a_1_1', 'b_1_1', 'c_1_1', 'd_1_1',  ]

xsymbol_list = ['0_1_1', '1_1_1', '2_1_1', '3_1_1',  '4_1_1',   
               '5_1_1', '6_1_1', '7_1_1', '8_1_1', '9_1_1'] 
#30/30 [==============================] - 17s 579ms/step - loss: 0.1625 - mean_squared_error: 0.0064 - acc: 0.9585

xsymbol_list = ['0_1_1', '1_1_1', '2_1_1', '3_1_1',  '4_1_1', '4_2_nose',  
               '5_1_1',  '5_2_hook', '6_1_1',  '7_1_1', '8_1_1', '9_1_1', 
               'a_1_1', 'b_1_1', 'c_1_1', 'd_1_1',  ]
#Epoch 200/200
#30/30 [==============================] - 23s 775ms/step - loss: 0.3869 - mean_squared_error: 0.0094 - acc: 0.9068

xsymbol_list= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",      
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",      
 "a_1_1",       "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "y_2_flower",  "x_2_left",   
 "x_2_right",   "x_1_1",       "\\sqrt_1_1",  "L_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",       "\\lt_1_1",   
 "p_1_1",       "p_2_ear",     "q_1_1",       "j_2_1",       "f_2_cobra",   "\\{_1_1",     "\\}_1_1",     "]_1_1",      
 "9_1_1",       "h_1_1",       "\\int_1_1",   "t_2_tail",    "e_1_1",       "z_1_1",       "g_1_1",       "s_1_1",      
 "5_2_hook",    "o_1_1",       "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",       "[_1_1",       "\\infty_1_1", "/_1_1"]      

#Epoch 907/2000
#30/30 [==============================] - 34s 1s/step - loss: 0.3563 - mean_squared_error: 0.0027 - acc: 0.8891


symbol_list= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",      
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",      
 "a_1_1",       "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "y_2_flower",  "x_2_left",   
 "x_1_1",       "\\sqrt_1_1",  "L_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",       "\\lt_1_1",   
 "p_1_1",       "p_2_ear",     "q_1_1",       "j_2_1",       "f_2_cobra",   "\\{_1_1",     "\\}_1_1",     "]_1_1",      
 "9_1_1",       "h_1_1",       "\\int_1_1",   "t_2_tail",    "e_1_1",       "z_1_1",       "g_1_1",       "s_1_1",      
 "5_2_hook",    "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",       "[_1_1",       "\\infty_1_1", "/_1_1"]      
removed_list = " : o_1_1   x_2_right"

#$ "\\{_1_1",     "\\}_1_1",     "]_1_1",    "[_1_1",    
#$
#$[_1_1, ]_1_1, (_1_1, x_2_right, L_1_1, p_2_ear, "\\{_1_1",     "\\}_1_1",

#save_file = './math_equation_1000.h5'
#https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
#confusion matrix generator


#file_ = './abc_head_essence_final_2013.csv'
file_ = './abc_test_2011_12_13.csv'

t = [] 
k = []
l = []
s = []
verbose = False
with open(file_, 'r') as f:
    #reader = csv.reader(f, delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
    reader = csv.reader(f, skipinitialspace=True) # delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
    next(reader, None)  #skipping header
    for row in reader:
        trace_regular =  row[13]
        key =  row[-1]

        #https://www.regular-expressions.info/floatingpoint.html
        refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", trace_regular)
        map_float = np.array( list( map(float, refindout)))
        strokes = np.reshape( map_float , (-1, 2))
        if( key in  symbol_list):
          t.append(strokes)
          k.append(key)
          l.append(len(strokes))
          s.append(row[0]) #sequence

        if(verbose):
          print(row)
          print(trace_regular)
          print(refindout)
          print( map_float)


t = np.asarray(t)

datasize = len(l)
print( " read %d strokes from %s " % (datasize, file_ ))
#print(l)

def TRACE_print( t ):
  for trace in t:
     print( "len:%d, first %s %s, last %s %s" % (len(trace), trace[0,0], trace[0,1], trace[-1,0],trace[-1,1]))
     print( trace.shape )
     
def zeropad ( t, timestep): # t is batch size of (N, 2) , N is number of points
  batch = len(t)
  #print( " batch %d , timestep %d " % (batch, timestep))
  #print(timestep)
  new_x = np.zeros((batch, timestep, 2))
  for i in range(0,batch):
     #print( len(t[i,]) )
     new_x[i,:len(t[i,])] = t[i,]
     #print(t[i,])
     #print(new_x[i,])
  return(new_x)

def findindex ( lib, keys ):
  idex = []
  #library = lib.tolist()
  for k in keys:
    #print(k)
    idex.append(lib.index(k))
  #print(idex)
  return(idex)


##sequence_length = 100'
##k_list = np.sort(k)
##print(k_list)
#k_set = np.sort(list(set( k ))) #using this set of symbols was not a good idea because RNN uses this k_set and I think it uses symbol_list --- prediction_generator was wrong!!!
#print("the key set list")
#xcnt = 0
#for i in np.arange(len(k_set)):
# print("%d %s"%(i, k_set[i]))
#
#
##idx = findindex( k_set, list(["a_1_1", "b_1_1", "z_1_1"]))
##print(idx)
##k_list = list(k_set)
#print( k_set )
#k_count = len(k_set)
#print(k_count)
#mask = np.array([[1,   1], [1,   1],[1,   1],[1,   1],[1,   1],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
#print (mask)
def train_generator():
  global current
  global datasize
  global t, k, l, s
  global verbose
#  global k_set, k_count
  global symbol_list
  current = 0
  print("run generator first time current: %d" % (current))

  while True:

        begin = current
        end = begin + batch_size
        if( end > datasize): 
            end = end - datasize
            x_train = np.append(t[begin:], t[:end])
            y_train = np.append(k[begin:], k[:end]) #k[begin:] + k[:end]
            l_train = np.append(l[begin:], l[:end])
            s_train = np.append(s[begin:], s[:end])            
        else:
            x_train = t[begin:end]
            y_train = k[begin:end]
            l_train = l[begin:end]
            s_train = s[begin:end]        

        print("run generator first time current: %d, end:%d" % (begin, end))
        

        current = end
        timestep = int( max(l_train) )

        x_train = zeropad(x_train, timestep) # np.zeros((timestep, 2))
#        idx = findindex( k_set,  y_train)
        idx = findindex( symbol_list,  y_train)

        category_y = to_categorical(idx , num_classes = len(symbol_list))
        #TRACE_print( x_train )
#        print(idx)
#        print(category_y)

        if(verbose):
          print( " begin %d , end %d " % (begin, end))
          print(x_train.shape)
          print( y_train )
          print(category_y)
 
        #print( l_train )
        if(verbose):
          print( s_train )
        #print( "max_time %d" % (timestep) )

##        if(i == 500*100):
##           print(i)
          print(x_train.shape)
##           print(x_train[i,])
##           print(y_train[i])

        yield x_train, category_y

#t_g = train_generator()
#t1, t2 = tee(t_g)

#for i in range(0,10):
#    print( " ------------------------------- %d"% i )
#    next(t_g)


#import numpy as np
#model = load_model('./me_ep1000_var61_acc90.h5') #./math_equation.h5', custom_objects = { '0':'a', '1':'b', '2': 'c', '3':'d'})
model = load_model('./model_sym61_batch500_epoch1500_1st64_2nd32.h5')

print(model)



#model = Sequential()
##model.add(LSTM(32, return_sequences=True, input_shape=(None, 2)))
##model.add(LSTM(8, return_sequences=False)) 
#
#model.add(LSTM(64, return_sequences=True, input_shape=(None, 2)))
##model.add(LSTM(16, return_sequences=False)) 
#model.add(LSTM(32, return_sequences=False)) 
#
##model.add((Dense(3, activation='sigmoid')))  
#model.add((Dense(k_count, activation='softmax')))  
#
#The sigmoid activation is outputing values between 0 and 1 independently from one another.
#If you want probabilities outputs that sums up to 1, use the softmax activation on your last layer, it will normalize the output to sum up to 1. 

#model.add((Dense(2, activation='softmax')))  

	#timedistributed requires all sequences (return_sequences = True). 
	#https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras

#model.compile(loss='categorical_crossentropy',
#                optimizer='adam',  metrics=['mse', 'accuracy'])
#    https://github.com/keras-team/keras/issues/2548
#    evaluate() will return the list of metrics that the model was compiled with. 
#    So if your compile metrics list include 'accuracy', then this should still work.

t1 = train_generator()
 
#model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)
num_batch_in_epoch = round(datasize / batch_size) + 1
print( "datasize %d" % (datasize ))
print( "batch_size %d" % (batch_size ))
print( "num_batch_in_epoch %f" % (num_batch_in_epoch ))
score  = model.evaluate_generator(t1, num_batch_in_epoch)#, verbose=1)
print( score ) 

#current = 0 #reset the location of generator -- this is bad practice of using a global variable
#I fixed this issue using current local variable and reset it outside of the whileloop
#generator stop at the yield and waits for the next call. On the next call, it loops at the while loop and current got never reset to zero again.
t2 = train_generator()

score_pre  = model.predict_generator(t2, num_batch_in_epoch)#, verbose=1)
print( score_pre ) 
print( "score_pre.shape ") 
print( score_pre.shape ) 
print( score_pre.shape[0] ) 

rr = np.arange(score_pre.shape[0])

print("len of k is %d" % (len(k)))
print("len of symbol_list is %d" % (len(symbol_list)))

for i in rr:
  e = k[ i] 
  sco = score_pre[i] #score
  p = symbol_list[ np.argmax(sco) ]
  if( e != p ):
    print( "DIFF %3d: exp=%10s and pre=%10s" % (i, e, p) )
  if( e == p ):
    print( "SAME %3d: exp=%10s and pre=%10s" % (i, e, p) )

  if( i == 337 ):
    print( sco )
    for j in np.arange(len(sco)):
      print ("%d score:%5f %10s"%(j, sco[j], symbol_list[j] ))


#sklearn is not installed and cannot be installed without causing confilct with existing python
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_train, score_pre)
#print(cm)


##     for i in np.arange(10):
##       sco = score_pre[i]
##       #print( sco )
##       #print( np.argmax(sco, axis = 0) )
##       e = k[ i] 
##       #print("k 0 %s 1 %s 2 %s"%(k[0], k[1], k[2]))
##       #pred_idx = k_count - np.argmax(sco, axis = 0) -1
##       pred_idx = np.argmax(sco)
##       p = symbol_list[ pred_idx ]
##       #print( "%3d: exp=%10s and pre=%10s max=%d" % (i, e, p, np.argmax(sco, axis = 0)) )
##       print( "%3d: exp=%10s and pre=%10s idx=%d" % (i, e, p, pred_idx) )



#https://datascience.stackexchange.com/questions/13894/how-to-get-predictions-with-predict-generator-on-streaming-test-data-in-keras
#from sklearn.metrics import confusion_matrix
#y_true = np.array([0] * 1000 + [1] * 1000)
#y_pred = probabilities > 0.5
#confusion_matrix(y_true, y_pred)


#model.fit_generator(train_generator(), steps_per_epoch=num_batch_in_epoch, epochs=30, verbose=1)
#history_callback = model.fit_generator(train_generator(), steps_per_epoch=30, epochs=epoch_size, verbose=1)
#print(history_callback.history.keys())

#loss_history = history_callback.history['loss']
#numpy_loss_history = np.array(loss_history)
#np.savetxt("./history_loss_1000.txt", numpy_loss_history, delimiter=",")

#acc_history = history_callback.history['acc']
#numpy_acc_history = np.array(acc_history)
#np.savetxt("./history_acc_1000.txt", numpy_acc_history, delimiter=",")

#mse_history = history_callback.history['mean_squared_error']
#numpy_mse_history = np.array(mse_history)
#np.savetxt("./history_mse_1000.txt", numpy_mse_history, delimiter=",")

#model.save(save_file)


# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# import numpy as np
# from tensorflow.python.keras.models import load_model
# model = load_model('/home/young/Tensorflow_projects/coding/keras_rnn_many_to_one_X.h5')
# #the batch size of 1 test sample
# x_test_up = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     1.1], [0.6,     1.3], [0.7, 1.5], [0.8,     1.7], [0.9,     1.9]]) 
# x_test_up = x_test_up.reshape(1,10,2)
# print( model.predict(x_test_up, batch_size=None, verbose=1) )
# #   array([[0.00663349, 0.6025158 ]], dtype=float32)   -> 0/1 , 1 is greater -> up sloping pattern

# x_test_decreasing  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .5], [0.6,  .4], [0.7,  .3], [0.8,  .2], [0.9,  .1]]) 
# x_test_decreasing= x_test_decreasing.reshape(1,10,2)
# model.predict(x_test_decreasing, batch_size=None, verbose=1)
# #   array([[0.5322375 , 0.00834433]], dtype=float32)   -> 0/1 , 0 is greater -> ----> correctly identifying it is decreasing!!!!

# x_test_down_up  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .6], [0.6,  .7], [0.7,  .8], [0.8,  .9], [0.9,  1]]) 
# x_test_down_up= x_test_down_up.reshape(1,10,2)
# model.predict(x_test_down_up, batch_size=None, verbose=1)
# #   array([[0.5102862 , 0.00811925]], dtype=float32)   -> 0/1 , 0 is greater ->-----> down and then up --- recognized as down

# x_test_up_down  = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     .6], [0.6,     .5], [0.7, .4], [0.8,     .3], [0.9,     .2]]) 
# x_test_up_down= x_test_up_down.reshape(1,10,2)
# model.predict(x_test_up_down, batch_size=None, verbose=1)
# #   array([[0.3975516, 0.011379 ]], dtype=float32)     -> 0/1 , 0 is greater ->---> this also down? 

# x_test_up_flat = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     0.9], [0.6,     0.9], [0.7, 0.9], [0.8,     0.9], [0.9,     0.9]]) 
# x_test_up_flat = x_test_up_flat.reshape(1,10,2)
# model.predict(x_test_up_flat, batch_size=None, verbose=1)
# #   array([[0.0125859 , 0.42021653]], dtype=float32)   -> 0/1 , 1 is greater -> recognized as up sloping
 




# Epoch 1/10
# 30/30 [==============================]30/30 [==============================] - 3s 103ms/step - loss: 1.0252 - mean_squared_error: 0.2066 - acc: 0.4854

# Epoch 2/10
# 30/30 [==============================]30/30 [==============================] - 2s 76ms/step - loss: 0.8104 - mean_squared_error: 0.1600 - acc: 0.7046

# Epoch 3/10
# 30/30 [==============================]30/30 [==============================] - 2s 75ms/step - loss: 0.5686 - mean_squared_error: 0.1050 - acc: 0.8064

# Epoch 4/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.2436 - mean_squared_error: 0.0361 - acc: 0.9540

# Epoch 5/10
# 30/30 [==============================]30/30 [==============================] - 2s 75ms/step - loss: 0.0628 - mean_squared_error: 0.0055 - acc: 0.9936

# Epoch 6/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.0356 - mean_squared_error: 0.0024 - acc: 0.9967

# Epoch 7/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.0251 - mean_squared_error: 0.0014 - acc: 0.9980

# Epoch 8/10
# 30/30 [==============================]30/30 [==============================] - 2s 76ms/step - loss: 0.0208 - mean_squared_error: 0.0012 - acc: 0.9981

# Epoch 9/10
# 30/30 [==============================]30/30 [==============================] - 2s 76ms/step - loss: 0.0176 - mean_squared_error: 9.7123e-04 - acc: 0.9983

# Epoch 10/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.0158 - mean_squared_error: 9.8142e-04 - acc: 0.9982

