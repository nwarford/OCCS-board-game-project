class myTree :
    
    def __init__ (self, path, attr):
        # For ID3, we need the attribute represented by the node
        # but also the value of the attribute that led us here
        # Path should be the empty string for the root
        # Attribute will be a label if it's a leaf
        self.data = [path, attr]
        self.children = []
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__ (self):
        return str(self.data) + ": " + str(self.children)
        
    def isRoot (self):
        if self.data[1] == None :
            return True
        else :
            return False
        
    def addChild (self, tree):
        self.children.append(tree)
        
    def getChild (self, attr):
        for child in self.children :
            if child.data == attr :
                return child
            
    def getAttr (self):
        return self.data[1]
    
    def getPath (self):
        return self.data[0]
    
    def isLeaf (self):
        if len(self.children) == 0 :
            return True
        else :
            return False