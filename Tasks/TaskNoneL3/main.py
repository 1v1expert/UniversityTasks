import numpy as np


class GameOnSphere(object):
    """Класс игры поиск на сфере
    point_number - количество точек 1 игрока
    epsilon - расстояние от 1 игрока, на котором 2 игрок
    считется найденным
    """
    def __init__(self, point_number=0, epsilon=0, radius=0):
        self.point_number = point_number
        self.radius = radius
        self.max_distance = epsilon / np.sin((np.pi - np.arcsin(epsilon / self.radius)) / 2)
        
    def get_point_on_sphere(self):
        z = np.random.uniform(-self.radius, self.radius)
        phi = np.random.uniform(0, 2 * np.pi)
        phi = 0 if phi == 2 * np.pi else phi
        r = np.sqrt(1 - z ** 2)
        x = np.cos(phi) * r
        y = np.sin(phi) * r
        return np.array([x, y, z])
    
    def get_points_for_player(self, name=""):
        points_list = []
        if name == "A":
            for i in range(self.point_number):
                points_list.append(self.get_point_on_sphere())
        elif name == "B":
            points_list.append(self.get_point_on_sphere())
            
        return points_list
    
    def win_function(self, player_a, player_b):
        win_a = False
        p_b = player_b[0]
        for p_a in player_a:
            distance = np.sqrt(np.sum((p_a - p_b) ** 2))
            if distance <= self.max_distance:
                win_a = True
        return win_a