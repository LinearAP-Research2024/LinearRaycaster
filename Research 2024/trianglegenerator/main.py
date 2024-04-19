import random

class Triangle:
    
    def __init__(self,p1,p2,p3,t1) -> None:
        self.verticies = [p1,p2,p3]
        self.target = t1
        self.intersect = self.checkIntersect()
    def checkIntersect(self):
        p = self.target
        p1 = self.verticies[0]
        p2 = self.verticies[1]
        p3 = self.verticies[2]
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(p, p1, p2) < 0.0
        b2 = sign(p, p2, p3) < 0.0
        b3 = sign(p, p3, p1) < 0.0

        return (b1 == b2) and (b2 == b3)
    def getIntersect(self) -> bool:
        return self.intersect


    

ret = "["
ret2 = ""
for i99 in range(500):
    #v = (np.random.rand(4,2) * 255).tolist()
    v = [[random.uniform(1,256) for _x in range(2)] for _i in range(3)]
    v.append([random.uniform(65,196) for _x in range(2)])
    #print(v)
    t = Triangle(v[0], v[1], v[2], v[3])
    ret += "[{},{},{},{},{}],\n".format(v[0],v[1],v[2],v[3],t.getIntersect())
    ret2 += "polygon(({},{}),({},{}),({},{}))({},{})\n".format(v[0][0],v[0][1],v[1][0],v[1][1],v[2][0],v[2][1],v[3][0],v[3][1])
ret+="]"
f = open("demofile3.txt", "w")
f.write(ret)
f.close()
f = open("demofile4.txt", "w")
f.write(ret2)
f.close()