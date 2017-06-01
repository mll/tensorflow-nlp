# Created by Marek Lipert (2017). All rights reserved.
# Can be distributed under GPLv3
# See the LICENSE file for details

import tensorflow as tf
import sys
from dataset import DataSet
import time
import train
from functools import reduce
import csv

def read_test_cases():
    cases = []
    ids = []
    with open('test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')        
        reader.next()
        for row in rdr:
           cases.append([row[1], row[2]])
           ids.append(row[0])
    return cases, ids

network = train.NeuralNetwork(train.conv_filters, train.filters_per_conv)
cases, ids = read_test_cases()

with open('submission.csv', 'w') as subm:
    subm.write('test_id,is_duplicate\n')
    all_predictions = []
    batch_size = 1000
    batch_count = int(len(cases)/batch_size) + 1
    for batch_no in range(0, batch_count):
        if (batch_no+1)*batch_size < len(cases):
            case = cases[batch_no*batch_size:(batch_no+1)*batch_size]
            indices = ids[batch_no*batch_size:(batch_no+1)*batch_size]
        else:
            case = cases[batch_no*batch_size:]
            indices = ids[batch_no*batch_size:]
        sys.stdout.write(str(batch_no)+"/"+str(batch_count)+"\r")
        sys.stdout.flush()

        predictions = map(lambda l: (l[1], 1 if l[0][0] > l[0][1] else 0),  zip(network.predict_for_sentence_pairs(case), indices)) 
        for prediction in predictions:
            all_predictions.append(prediction)
    
    for prediction in all_predictions:
        subm.write(str(prediction[0])+","+str(prediction[1])+"\n")
    