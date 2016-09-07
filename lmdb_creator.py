import numpy as np
import lmdb
import caffe
from caffe.proto import caffe_pb2


"""

following code takes a lmdb database and the labels as inputs then merges them and
creates a new dataset and outputs it

lmdb database stores the images in the order they appear in nautilus

the labels should contain only the class values, not the names of the images

"""

# label values stroed in column wise fashion
label = 'cars_and_bikes.txt'


# the following is the features extracted without their labels
file = '/home/ali/caffe/training_features'
env = lmdb.open(file)
# following creates an array full of label values

labels = []
cursor = env.begin().cursor()

with open(label) as f:
    for line in f:
        labels.append(int(line))
i = 0
datum = caffe_pb2.Datum()
"""
# prints the labels

with env.begin(write='True') as txn:
    for i in range(len(labels)):
        # the keys are all 10 digist with preceding 0's
        keystr = '0'*(10-(len(str(i))))+str(i)
        buf = txn.get(keystr)
        datum.ParseFromString(buf)
        print datum.label
        i = i+1
"""
"""

# following is to check if accidentally some distortions occured

with env.begin() as txn:
 i=20   
 # the keys are all 10 digist with preceding 0's
 keystr = '0'*(10-(len(str(i))))+str(i)
 buf = txn.get(keystr)
 datum.ParseFromString(buf)
 print buf
 print '\n*********************************\n'


env = lmdb.open('/home/ali/caffe/cars_and_bikes_features')

with env.begin() as txn:
 i=20   
 # the keys are all 10 digist with preceding 0's
 keystr = '0'*(10-(len(str(i))))+str(i)
 buf = txn.get(keystr)
 datum.ParseFromString(buf)
 print buf
 print '\n'
"""


# 
# The actual code that merges labels with extracted features
#
with env.begin(write=True, buffers=True) as txn:

    for i in range(len(labels)):
        # the keys are all 10 digist with preceding 0's
        keystr = '0'*(10-(len(str(i))))+str(i)
        buf = txn.get(keystr)
        txn.delete(keystr)
        buf_copy = bytes(buf)

        datum.ParseFromString(buf)
        datum.label = labels[i]
        

        txn.put(keystr, datum.SerializeToString())


 ## check if the ouputs are all the same
 ## this was the case for prior absurdity
"""
with env.begin() as txn:
    a = txn.get("0000000035")
    b = txn.get("0000000006") 
    print a
    datum1 = caffe_pb2.Datum()
    datum2 = caffe_pb2.Datum()
    datum1.ParseFromString(a)
    datum2.ParseFromString(b)
    a1 = caffe.io.datum_to_array(datum1)
    b1 = caffe.io.datum_to_array(datum2)
    print np.array_equal(a1, b1)
    print "***********"

"""
    
"""
TODO:
   cursor = txn.cursor()
   for key, value in cursor:
       print(key)

- make it a function that takes command line arguments as parameter.

"""
 ## prints the keys 
"""
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
       print(key)
"""
