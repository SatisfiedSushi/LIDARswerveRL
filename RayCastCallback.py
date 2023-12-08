from Box2D.Box2D import *


class RayCastCallback(b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.m_fixture = None
        self.m_point = b2Vec2()
        self.m_normal = b2Vec2()
        self.m_fraction = 0.0

    def ReportFixture(self, fixture, point, normal, fraction):
        #print(point)
        self.m_fixture = fixture
        self.m_point.Set(point[0], point[1])
        self.m_normal.Set(normal[0], normal[1])
        self.m_fraction = fraction
        return fraction
