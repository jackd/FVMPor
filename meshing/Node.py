import meshDefs

class Node:
    def __init__(self,coords=[0, 0, 0],props=[]):
        self.coords=coords
        self.props=props
    def getCoords(self):
        return self.coords
    def getProps(self):
        return self.props


if __name__ == "__main__":
    n = Node([0,1,0],[-2])
    print n.getCoords()
    print n.getProps()
