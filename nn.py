import sys, random, csv
import tensorflow as tf

def parseData(filename) :
    # Args:
    # filename (str) : the name of the file
    # one_hot (bool) : true if we're using one hot coding (for monks)
    # Constructing our full set of instances from the .csv
    instanceList = []
    labelList = [] # We also need to know what our possible labels are,
    # to go back and make individual label lists for every instance
        
    # Processing the .csv
    with open(filename, 'r') as csvFileIn :
        # is there a way to see if this worked?
        dataset = csv.reader(csvFileIn) 
        
        skipFirstRow = True;
        i = 0 # our counter for the number of instances
        for row in dataset :
            if skipFirstRow :
                skipFirstRow = False
                attrRow = row[1:] # need to store the labels
            else :
                instanceList.append([])
                instanceList[i].append(row) # our instance list is one shorter than our data,
                # and each instance needs to have two things in it. We'll go back
                # later and add labels as the second item in each instance
                if not(row[0] in labelList) :
                    # If we've encountered a new label, add it! hopefully not too slow, because 
                    # our list of labels is probably not that long
                    labelList.append(row[0])
                i += 1
                
    # must determine if it's binary
    isBinary = False
    if len(labelList) == 2 :
        isBinary = True
        
    # Next, we need to slice off the label of each attribute list, and add a label list for each item
    for i in range(len(instanceList)) :
        currRow = instanceList[i][0]
        label = currRow[0]
        labelIndex = labelList.index(label) # Fetching the label's global index
        specificList = [] # This is our individualized label list for the instance
        if not(isBinary) :
            for j in range(len(labelList)) :
                if j == labelIndex :
                    specificList.append(1)
                else :
                    specificList.append(0)
        else :
            if label == "yes" :
                specificList.append(1)
            else :
                specificList.append(0)
        
        instanceList[i][0] = instanceList[i][0][1:] # Slicing off the label once we've used it
        instanceList[i].append(specificList)
        
    
    return [instanceList, attrRow, labelList]

def oneHot(instanceList, labels) :
    # Args:
    # instanceList (list) : an array of instances, usually monks1.csv in our testing
    # labels (list) : an array of our labels
    
    # Strategy = go through every item's attribute
    # For each attribute,
    #   if it's new,
    #       add it to our label tracker (length of labels, with an array for each label)
    #   Set the instance's appropriate bit to 1
    # Once this is all done, remove the last bit from each instance for each attribute
    
    labelTracker = []
    for _ in labels :
        labelTracker.append([])
    
    # First, count how many values there are for each attribute
    for instance in instanceList : 
        for i in range(len(instance[0])) :
            attribute = instance[0][i]
            if not(attribute in labelTracker[i]) :
                labelTracker[i].append(attribute)
    
    # AFTER we've done that, reconstruct our instances
    for i in range(len(instanceList)) :
        instance = instanceList[i]
        newInstance = [] # We need to replace the old instance with the new one
        for j in range(len(instance[0])) : # For every attribute...
            value = instance[0][j] # get the value
            valueIndex = labelTracker[j].index(value) # find its corresponding index out of that attribute
            for k in range(len(labelTracker[j]) - 1) : # for all possible values of that attribute... 
                if k == valueIndex : # if it's the same index, add a 1
                   newInstance.append(1)
                else : # otherwise, add a 0
                    newInstance.append(0)
            
        instanceList[i][0] = newInstance # replace the old instance with our new cool instance 
    
    return instanceList

def floatCast(instances) :
    # helper method to cast all of our attributes as floats
    for i in range(len(instances)) :
        row = instances[i][0]
        newRow = []
        for attr in row :
            newRow.append(float(attr))
            
        instances[i][0] = newRow
        
    return instances
    
    # instances = tf.convert_to_tensor(instances, dtype=tf.variant)
    # return tf.string_to_number(instances, dtype=tf.float32)

# helper method to extract the attributes
def getAtts(instances, retTensor):
    # Args:
    # instances (list) : an array of instances, where an instance is a list of 2 lists: attributes, labels
    # retTensor (boolean) : boolean value to return extracetd values as either a list or a tensor
    
    attList = []
    for i in instances:
        attList.append(i[0])
    if retTensor:
        attTensor = tf.convert_to_tensor(attList, preferred_dtype=tf.float32)
        return attTensor
    else:
        return attList

# helper method to extract the labels
def getLabels(instances, retTensor):
    # Args:
    # instances (list) : an array of instances, where an instance is a list of 2 lists: attributes, labels
    # retTensor (boolean) : boolean value to return extracetd values as either a list or a tensor
    
    labelList = []
    for i in instances:
        labelList.append(i[1])
    if retTensor:
        labelTensor = tf.convert_to_tensor(labelList, preferred_dtype=tf.float32)
        return labelTensor
    else:
        return labelList

# Helper method to get the max item in a list
def maxIndex(list) :
    max = None
    for i in range(len(list)) :
        if max == None :
            max = list[i]
        elif list[i] > max :
            max = list[i]
    return list.argmax()

# TODO: Testing
# tensors dtype = float32 or float64; normal default is int32
def networkModel(trainInstances, neurons):
    hidden_layer_size = neurons
    input_layer_size = len(trainInstances[0][0])
    output_layer_size = len(trainInstances[0][1])
    num_instances = len(trainInstances)
    
    global func
    if output_layer_size == 1:
        func = 0
    else:
        func = 1
        
    hidden_layer = {'weights':tf.Variable(tf.truncated_normal([input_layer_size, hidden_layer_size], stddev=0.01)),
                    'biases':tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.01))}
    
    output_layer = {'weights':tf.Variable(tf.truncated_normal([hidden_layer_size, output_layer_size], stddev=0.01)),
                    'biases':tf.Variable(tf.truncated_normal([output_layer_size], stddev=0.01))}
    
    
    layer1 = tf.add(tf.matmul(x, hidden_layer['weights']), hidden_layer['biases'])
    layer1 = tf.nn.sigmoid(layer1)
    
    output = tf.add(tf.matmul(layer1, output_layer['weights']), output_layer['biases'])
    if func == 0:
        output = tf.nn.sigmoid(output)
    else:
        output = tf.nn.softmax(output)
    
    return output
    
def trainNetwork(model, learnRate, iterations, trainInstances, testInstances):
    # Args:
    # model : neural network we setup earlier
    # learnRate : learning rate, as a parameter from the original call
    # iterations : how many times we learn from the training data
    # trainInstances : our training set
    # testInstances : our test set
    if func == 0:
        cost = tf.reduce_sum(0.5 * ((y - model) * (y - model)))
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)
    
    trainAtts = getAtts(trainInstances, False)
    trainLabels = getLabels(trainInstances, False)
    testAtts = getAtts(testInstances, False)
    testLabels = getLabels(testInstances, False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # do we need this?
        
        for i in range(iterations):
            u, loss = sess.run([optimizer, cost], feed_dict={x : trainAtts, y : trainLabels})
            
            if (i % (iterations // 20) == 0) :
                print("Iteration " + str(i) + " of " + str(iterations) + ", loss = " + str(loss))
                
            # Block for observing learning rate differences
            if (i > 0) and (i % 50 == 0):
                p = sess.run(model, feed_dict={x: testAtts, y: testLabels})
                total = len(testLabels)
                accNum = 0
                for k in range(len(testLabels)) :
                    predIndex = None
                    if len(labelList) > 2 :
                        labelIndex = testLabels[k].index(1)
                        predIndex = p[k].argmax()
                    else : # accounting for binary here, I think
                        labelIndex = testLabels[k][0]
                        if p[k] >= .5 :
                            predIndex = 1
                        else :
                            predIndex = 0
                        
                    if labelIndex == predIndex :
                        accNum += 1
                
                p = sess.run(model, feed_dict={x: trainAtts, y: trainLabels})
                total2 = len(trainLabels)
                accTrain = 0
                for k in range(len(trainLabels)) :
                    predIndex = None
                    if len(labelList) > 2 :
                        labelIndex = trainLabels[k].index(1)
                        predIndex = p[k].argmax()
                    else : # accounting for binary here, I think
                        labelIndex = trainLabels[k][0]
                        if p[k] >= .5 :
                            predIndex = 1
                        else :
                            predIndex = 0
                        
                    if labelIndex == predIndex :
                        accTrain += 1
                
                print("Test Set Accuracy: " + str(accNum/total) + "\nTraining Set Accuracy: " + str(accTrain/total2))
                # End block
        
        # run predictions on the test set
        # p is a 2D array - first index is the instance, second instance is the probability
        # for the label, with this index corresponding to labelList
        p = sess.run(model, feed_dict={x: testAtts, y: testLabels})        
    
    return [p, testLabels]
    

if len(sys.argv) != 7 :
    raise Exception("Incorrect input arguments: expecting 6 inputs")
# Basic run arguments, provided by the run button: mnist_1000.csv 20 0.001 1000 0.75 12345

filename = sys.argv[1]

neurons = int(sys.argv[2])
learning_rate = float(sys.argv[3])
iterations = int(sys.argv[4])
percent = float(sys.argv[5]) # Percent that divides training and test data
if percent <= 0 or percent >= 1 :
    raise Exception("Percent must be between 0 and 1")
seed = int(sys.argv[6])
random.seed(seed)
tf.set_random_seed(seed)  # controls random start weights and biases in nn model
func = 1  # use values as {0=sigmoid function} and {1=softmax function}

[instanceList, attrList, labelList] = parseData(filename) # having the labels makes one-hotting easier

if filename == "monks1.csv" :
    instanceList = oneHot(instanceList, attrList)

instanceList = floatCast(instanceList) # Gotta cast those floats

# Splitting into training and test sets
random.shuffle(instanceList)

trainingIndex = round(len(instanceList) * percent)
trainingSet = []
for i in range(trainingIndex) :
    # Fetch all items up to trainingIndex for our training set
    trainingSet.append(instanceList[i])

testSet = []
for j in range(trainingIndex, len(instanceList)) :
    testSet.append(instanceList[j])

input_layer_size = len(trainingSet[0][0])
output_layer_size = len(trainingSet[0][1])
x = tf.placeholder(tf.float32, [None, input_layer_size])
y = tf.placeholder(tf.float32, [None, output_layer_size])
m = networkModel(trainingSet, neurons)
[probArray, testLabels] = trainNetwork(m, learning_rate, iterations, trainingSet, testSet)

# Here, we need to generate our confusion matrix, which works by comparing our probabilities in probArray to our testLabels

# make the matrix whose information we'll put in the csv here
matrixInfo = []
for i in range(len(labelList)) :
    rowList = []
    for j in range(len(labelList)) :
        rowList.append(0)
    matrixInfo.append(rowList)
    
# TODO: actually make predictions
# Will involve using maxIndex on our probArray elements and comparing that to testLabels
total = len(testLabels)
accNum = 0
for i in range(len(testLabels)) :
    predIndex = None
    if len(labelList) > 2 :
        labelIndex = testLabels[i].index(1)
        predIndex = probArray[i].argmax()
    else : # accounting for binary here, I think
        labelIndex = testLabels[i][0]
        if probArray[i] >= .5 :
            predIndex = 1
        else :
            predIndex = 0
        
    if labelIndex == predIndex :
        accNum += 1
    
    matrixInfo[labelIndex][predIndex] += 1

print("Accuracy : " + str(accNum / total))
print("Numerator : " + str(accNum))
print("Denominator : " + str(total))

    
endIndex = len(sys.argv[1]) - 4
fileLessCSV = sys.argv[1][:endIndex]
outputFile = "results_" + fileLessCSV + "_" + str(neurons) + "n_" + str(learning_rate) + "r_" + str(iterations) + "i_" + str(percent) + "p_" + str(seed) + ".csv" # filename
with open(outputFile, "w") as confMatrix :
    matrixWriter = csv.writer(confMatrix)
    
    matrixWriter.writerow(labelList)
    
    for i in range(len(matrixInfo)) :
        matrixInfo[i].append(labelList[i])
        matrixWriter.writerow(matrixInfo[i])



    
