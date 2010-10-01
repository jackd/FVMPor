import meshDefs

class Element:
    def __init__(self,type,nodes,elementProps=[0],faceProps=[]):
        (nNodes, nFaces, order, name) = meshDefs.elementDefs[type]
        self.type=type
        self.nodes=nodes
        self.elementProps=elementProps
        if faceProps==[]:
            for x in range(nFaces):
                faceProps.append(0)
        self.faceProps = faceProps

    def getElementProps(self):
        return self.elementProps
    def getType(self):
        return self.type
    def getNodes(self):
        return self.nodes
    def getFaceProps(self):
        return self.faceProps
    def setNodes(self,newNodes):
        self.nodes = newNodes
    def setFaceProps(self,newFaceProps):
        self.faceProps = newFaceProps

    # tests if passed element is a face element of this element
    def isFace(self,fElement):
        return false;
