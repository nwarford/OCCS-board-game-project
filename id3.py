import sys, csv, random, math
from customTree import myTree

# FUNCTION DEFINITIONS

def sameLabel (instances):
    # Returns true if all attributes in instances are the same
    firstItem = instances[0][0]
    for item in instances :
        if item[0] != firstItem :
            return False
    return True

def gain (instances, attr):
    # Will return the gain of the set INSTANCES when sorted by attribute ATTR
    values = list(attrValues[attr]) # values is the set of possible answers
    attrIndex = attrRow.index(attr) # Where to find it in the instance
    
    subsets = []
    for i in range(len(values)) :
        subsets.append([]) #indices in subsets correspond to indices in values
    
    for instance in instances :
        currAttr = instance[attrIndex]
        currIndex = values.index(currAttr)
        subsets[currIndex].append(instance)
    
    gainSum = entropy(instances)
    denominator = len(instances)
    for subset in subsets :
        numerator = len(subset)
        if len(subset) > 0: # Need not do this if there isn't a subset to evaluate
            gainSum -= (numerator / denominator) * entropy(subset)
        
    return [gainSum, subsets, values]
    
def entropy(instances):
    # Returns the entropy of the set based on the labels provided
    denominator = len(instances)
    countList = []
    for label in labelList :
        countList.append(0)
    
    for instance in instances :
        index = labelList.index(instance[0])
        countList[index] += 1
    
    sumTotal = 0
    for count in countList :
        fraction = count / denominator
        if fraction != 0 :
            sumTotal += (fraction) * math.log(fraction, 2)
        
    return -1 * sumTotal

def maxLabel (instances):
    # Returns the label from labelList that most frequently appears in instances
    countList = []
    for i in range(len(labelList)) :
        countList.append(0)
    
    for instance in instances :
        label = instance[0]
        index = labelList.index(label)
        countList[index] += 1
        
    return labelList[countList.index(max(countList))] # Return the label of the highest 

# ID3 ALGORITHM
def id3 (attributes, instances, pathto):
    # Uncomment print statements in here to see what exactly the algorithm is doing
    if len(attributes) == 0 :
        #print("Make a leaf with the most common label")
        commonLabel = maxLabel(instances)
        returnTree = myTree(pathto, commonLabel)
    elif sameLabel(instances) :
        #print("Make a leaf with the label these instances share")
        returnTree = myTree(pathto, instances[0][0])
    else :
        # The hard part
#         print("Do ID3 in proper")
        # Find the best attribute 
        maxGain = 0
        bestGain = None # The current best gain-yielding attribute
        bestSets = None # The sets, divided by this attribute
        bestValues = None
        for attribute in attributes :
            [attrGain,subsets,values] = gain(instances, attribute)
            if attrGain >= maxGain :
                # When we find our highest entropy, set what that gain is...
                maxGain = attrGain
                # what attribute got it...
                bestGain = attribute
                # what the subset split based on that data is...
                bestSets = subsets
                # and what the values for that attribute are.
                bestValues = values
        
        returnTree = myTree(pathto, bestGain)
        
        for value in bestValues :
            valueSet = bestSets[bestValues.index(value)]
            if len(valueSet) == 0 : # Do our best guess!
                bestGuess = maxLabel(instances)
                returnTree.addChild(myTree(value, bestGuess))
            else : 
                # Remove the attribute with incomplete deep copy
                newAttributes = []
                for attr in attributes :
                    if attr != bestGain :
                        newAttributes.append(attr)
                if len(valueSet) == 0 :
                    #print("Value set empty")
                    # If there are no values in the attribute, return the highest occurring label in the current set
                    childTree = myTree(pathto, maxLabel(instances))
                else :
                    childTree = id3(newAttributes, valueSet, value)
                returnTree.addChild(childTree)
                
    
    return returnTree

def evalInstance (instance, tree):
    currNode = tree
    
    while not(currNode.isLeaf()) :
        currAttr = currNode.getAttr() # What attribute does this node evaluate?
        attrIndex = attrRow.index(currAttr) # Where is that stored in the instance?
        instanceVal = instance[attrIndex] # What's the value for this instance?
        for child in currNode.children :
            if child.getPath() == instanceVal : # The path that leads to the node is the *value* of the attribute
                currNode = child
        
    return currNode.getAttr() # When it's a leaf, we have an answer! Recursion is fun.

if len(sys.argv) != 4 :
    raise Exception("Incorrect input arguments: expecting 3 inputs")
# Basic run arguments, provided by the run button: iris.csv h 1 0.5 12345
    
# getting our basic arguments here
percent = float(sys.argv[2])
if percent <= 0 or percent > 1 :
    raise Exception("Percent must be between 0 and 1")
seed = int(sys.argv[3])
random.seed(seed)

# Constructing our full set of instances from the .csv
instanceList = []
labelList = [] # We also need to know what our possible labels are
attrValues = dict() # We are keeping a dictionary of possible values for each attribute. Used for gain

# Processing the .csv
with open(sys.argv[1], 'r') as csvFileIn :
    # is there a way to see if this worked?
    dataset = csv.reader(csvFileIn) 
    
    skipFirstRow = True;
    for row in dataset :
        if skipFirstRow :
            skipFirstRow = False
            attrRow = row # need to store the attributes
            for i in range(1, len(attrRow)) :
                attrValues[attrRow[i]] = set() # Using an empty set for now, skipping the label
        else :
            instanceList.append(row) # our set of instances is just an array of arrays
            if not(row[0] in labelList) :
                # If we've encountered a new label, add it! hopefully not too slow, because 
                # our list of labels is probably not that long
                labelList.append(row[0])
            # After we've done this, add possible attribute values
            for i in range(1,len(row)) :
                attrValues[attrRow[i]].add(row[i]) # Set takes care of dupes

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

workAttrRow = []
for i in range(1, len(attrRow)) :
    workAttrRow.append(attrRow[i]) # The first one is the name of the label
    
workInstanceList = []
for instance in trainingSet :
    workInstanceList.append(instance)

# Create our tree!
id3Tree = id3(workAttrRow, workInstanceList, None)
# print(id3Tree)
# make the matrix whose information we'll put in the csv here
matrixInfo = []
for i in range(len(labelList)) :
    rowList = []
    for j in range(len(labelList)) :
        rowList.append(0)
    matrixInfo.append(rowList)
# matrix layout: label0, label1, ... labeln-1 left to right and up and down
# Thus, top right is 0,0, bottom right is n-1,n-1
total = 0
correct = 0

for instance in testSet :
    prediction = evalInstance(instance, id3Tree)
    actual = instance[0]
    predIndex = labelList.index(prediction)
    actualIndex = labelList.index(actual)
    matrixInfo[actualIndex][predIndex] += 1
    # Counting the totals and correct answers to make processing later easier
    total += 1
    if actual == prediction :
        correct += 1
    

# Create the CSV output
endIndex = len(sys.argv[1]) - 4
fileLessCSV = sys.argv[1][:endIndex]
outputFile = "results_" + fileLessCSV + "_" + str(seed) + ".csv" # filename
with open(outputFile, "w") as confMatrix :
    matrixWriter = csv.writer(confMatrix)
    
    matrixWriter.writerow(labelList)
    
    for i in range(len(matrixInfo)) :
        matrixInfo[i].append(labelList[i])
        matrixWriter.writerow(matrixInfo[i])
    
    totalsRow = ["Total", total, "Correct", correct]
    matrixWriter.writerow(totalsRow)
    
    