import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



class ObservationTools():

    def __init__(self):
        self.grayscale_image = None
        self.ray_origin_x = 0
        self.ray_origin_y = 0
        self.size_x = None
        self.size_y = None
        self.forward_distance = 0
        self.left_distance = 0
        self.right_distance = 0
        self.forward_left_distance = 0
        self.forward_left_distance_2 = 0
        self.forward_left_distance_3 = 0
        self.forward_right_distance = 0
        self.forward_right_distance_2 = 0
        self.forward_right_distance_3 = 0
        self.ray_count = 9
        self.observation = np.zeros(self.ray_count, dtype=np.float32)
        self.old_observation = np.zeros(self.ray_count, dtype=np.float32)

        self.longest_measured_distance = 0
        self.total_measured_distance = 0
        self.longest_normalized_distance = 0
        self.total_normalized_distance = 0
        self.reward = 0
        self.old_reward = 0


    def visualize_grayscale_image(self):
        plt.imshow(self.grayscale_image, cmap='gray')
        plt.show(block=False)
        plt.pause(0.00001)
        plt.clf()


    def create_grayscale_image(self, observation, resizeX=None, resizeY=None, cutY=None):
        o = observation.copy()
        o[o[:,:,0] > 128] = 255
        if resizeX is not None or resizeY is not None:
            ro = Image.fromarray(o).resize((resizeX, resizeY))
        else:
            ro = Image.fromarray(o)
        gsi = np.asarray(ro.convert('L'), dtype=np.float32)
        if cutY is not None:
            gsi = gsi[:cutY,:]
        gsi = gsi / 255.
        gsi[gsi < 0.5] = 0.
        gsi[gsi >= 0.5] = 1.
        self.grayscale_image = gsi
        self.size_x = self.grayscale_image.shape[0]
        self.size_y = self.grayscale_image.shape[1]


    def calculate_rays(self, ray_origin_x, ray_origin_y, visualize=True, normalisation_factor = 70.):
        self.old_observation = self.observation.copy()
        self.ray_origin_x = ray_origin_x
        self.ray_origin_y = ray_origin_y

        self._left_ray(visualize=visualize)
        self.observation[0] = self.left_distance

        self._forward_left_ray(visualize=visualize)
        self.observation[1] = self.forward_left_distance

        self._forward_left_2_ray(visualize=visualize)
        self.observation[2] = self.forward_left_distance_2

        self._forward_left_3_ray(visualize=visualize)
        self.observation[3] = self.forward_left_distance_3

        self._forward_ray(visualize=visualize)
        self.observation[4] = self.forward_distance

        self._forward_right_3_ray(visualize=visualize)
        self.observation[5] = self.forward_right_distance_3

        self._forward_right_2_ray(visualize=visualize)
        self.observation[6] = self.forward_right_distance_2

        self._forward_right_ray(visualize=visualize)
        self.observation[7] = self.forward_right_distance

        self._right_ray(visualize=visualize)
        self.observation[8] = self.right_distance

        m = np.sum(self.observation)
        if m > self.total_measured_distance:
            self.total_measured_distance = m

        m = np.max(self.observation)
        if m > self.longest_measured_distance:
            self.longest_measured_distance = m

        self.observation = self.observation / normalisation_factor

        n = np.sum(self.observation)
        if n > self.total_normalized_distance:
            self.total_normalized_distance = n

        n = np.max(self.observation)
        if n > self.longest_normalized_distance:
            self.longest_normalized_distance = n
        

    def _forward_ray(self, visualize=True):
        self.forward_distance = 0
        for i in reversed(range(self.ray_origin_x - 4)):
            if self.grayscale_image[i, self.ray_origin_y] == 0:
                self.forward_distance += 1
                if visualize:
                    self.grayscale_image[i, self.ray_origin_y] = 1.
            else:
                break


    def _left_ray(self, visualize=True):
        self.left_distance = 0
        m = 0
        for i in reversed(range(self.ray_origin_y - 1)):
            if i % 3 == 0:
                m += 1
            if self.grayscale_image[self.ray_origin_x - m, i] == 0:
                self.left_distance += 1
                if visualize:
                    self.grayscale_image[self.ray_origin_x - m, i] = 1.
            else:
                break


    def _right_ray(self, visualize=True):
        self.right_distance = 0
        m = 0
        for i in range(self.ray_origin_y + 1, self.size_y):
            if i % 3 == 0:
                m += 1
            if self.grayscale_image[self.ray_origin_x - m, i] == 0:
                self.right_distance += 1
                if visualize:
                    self.grayscale_image[self.ray_origin_x - m, i] = 1.
            else:
                break


    def _forward_left_ray(self, visualize=True):
        self.forward_left_distance = 0
        for i in range(1, min(self.ray_origin_x, self.ray_origin_y) - 1):
            if self.grayscale_image[self.ray_origin_x - i, self.ray_origin_y - 1 - i] == 0:
                self.forward_left_distance += 1
                if visualize:
                    self.grayscale_image[self.ray_origin_x - i, self.ray_origin_y - 1 - i] = 1.
            else:
                break


    def _forward_right_ray(self, visualize=True):
        self.forward_right_distance = 0
        for i in range(1, min(self.ray_origin_x, self.ray_origin_y)):
            if self.grayscale_image[self.ray_origin_x - i, self.ray_origin_y + i] == 0:
                self.forward_right_distance += 1
                if visualize:
                    self.grayscale_image[self.ray_origin_x - i, self.ray_origin_y + i] = 1.
            else:
                break


    def _forward_right_2_ray(self, visualize=True):
        self.forward_right_distance_2 = 0
        y = self.ray_origin_y + 2
        for i in reversed(range(self.ray_origin_x - 2)):
            if i % 2 == 0:
                y += 1
            if y < self.size_y:
                if self.grayscale_image[i, y] == 0:
                    self.forward_right_distance_2 += 1
                    if visualize:
                        self.grayscale_image[i, y] = 1.
                else:
                    break
            else:
                break


    def _forward_left_2_ray(self, visualize=True):
        self.forward_left_distance_2 = 0
        y = self.ray_origin_y - 2
        for i in reversed(range(self.ray_origin_x - 2)):
            if i % 2 == 0:
                y -= 1
            if y > 0:
                if self.grayscale_image[i, y - 1] == 0:
                    self.forward_left_distance_2 += 1
                    if visualize:
                        self.grayscale_image[i, y - 1] = 1.
                else:
                    break
            else:
                break


    def _forward_right_3_ray(self, visualize=True):
        self.forward_right_distance_3 = 0
        y = self.ray_origin_y + 2
        for i in reversed(range(self.ray_origin_x - 5)):
            if i % 4 == 0:
                y += 1
            if y < self.size_y:
                if self.grayscale_image[i, y] == 0:
                    self.forward_right_distance_3 += 1
                    if visualize:
                        self.grayscale_image[i, y] = 1.
                else:
                    break
            else:
                break


    def _forward_left_3_ray(self, visualize=True):
        self.forward_left_distance_3 = 0
        y = self.ray_origin_y - 2
        for i in reversed(range(self.ray_origin_x - 5)):
            if i % 4 == 0:
                y -= 1
            if y > 0:
                if self.grayscale_image[i, y] == 0:
                    self.forward_left_distance_3 += 1
                    if visualize:
                        self.grayscale_image[i, y] = 1.
                else:
                    break
            else:
                break


    def debug_distances(self):
        print(self.left_distance, self.forward_left_distance, self.forward_distance, self.forward_right_distance, self.right_distance)
