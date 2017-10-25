import numpy as np
from math import cos, sin, radians, atan2

class CoordinateTransforms:

    def __init__(self):
        self.earth_reciprocal_flattening = 298.25
        self.earth_semimajor = 6378137.0  # m
        self.earth_semiminor = self.earth_semimajor * (1 - 1/self.earth_reciprocal_flattening)  # m
        self.earth_eccentricity = np.sqrt((self.earth_semimajor ** 2 - self.earth_semiminor ** 2) / (self.earth_semimajor ** 2))
        self.earth_omega = 7292115e-11  # rad/s
        self.c = 299792458.0  # m/s

    def geodetic_to_ecef(self, geodetic_array):
        geodetic_array = geodetic_array.reshape((-1,3))
        g_lat, g_lon, height = geodetic_array[:,0], geodetic_array[:,1], geodetic_array[:,2]
        g_lat, g_lon = np.radians(g_lat), np.radians(g_lon)
        N, N1 = self.calc_N_and_N1(g_lat)
        cosLat, cosLon = np.cos(g_lat), np.cos(g_lon)
        sinLat, sinLon = np.sin(g_lat), np.sin(g_lon)
        x = np.multiply((N + height), np.multiply(cosLat, cosLon))
        y = np.multiply((N + height), np.multiply(cosLat, sinLon))
        z = np.multiply((N1 + height), sinLat)
        output_array = np.stack((x, y, z), axis=1).reshape((-1,3)).squeeze()
        return output_array

    def ecef_to_geodetic(self, ecef_array):
        ecef_array = ecef_array.reshape((-1,3))
        output_array = np.copy(ecef_array)
        for index, row in enumerate(ecef_array):
            x, y, z = row[0], row[1], row[2]
            phi_guess = radians(45)
            N, N1 = self.calc_N_and_N1(phi_guess)
            h_guess = (np.sqrt(x**2 + y**2)/cos(phi_guess)) - N
            h_difference = 1.0
            while h_difference > .01:
                # I chose to iterate to close the height gap since the phi gap closed much more quickly
                # the height difference was the limiting factor in the iteration
                phi = np.arcsin(z /(N1 + h_guess))
                N, N1 = self.calc_N_and_N1(phi)
                h = (np.sqrt(x ** 2 + y ** 2) / cos(phi)) - N
                h_difference = abs(h - h_guess)
                phi_guess = phi
                h_guess = h
            g_lat = np.rad2deg(phi_guess)
            g_lon = np.rad2deg(atan2(y,x))
            output_array[index,:] = np.array([g_lat, g_lon, h])
        return output_array.reshape((-1,3)).squeeze()

    def ecef_to_enu(self, reference_ecef, object_ecef):
        reference_ecef, object_ecef = reference_ecef.reshape((-1,3)), object_ecef.reshape((-1,3))
        ref_geo = self.ecef_to_geodetic(reference_ecef)
        ref_lat, ref_lon = ref_geo[0], ref_geo[1]
        R = self.rotation_matrix_around_z_then_x(ref_lon, ref_lat)
        output_enu_array = np.dot(R, (object_ecef - reference_ecef).transpose())
        return output_enu_array.reshape((-1,3)).squeeze()

    def ecef2sky(self, reference_ecef, object_ecef):
        enu_array = self.ecef_to_enu(reference_ecef, object_ecef)
        e, n, u = enu_array[:,0], enu_array[:,1], enu_array[:,2]
        azimuth = np.arctan2(e,n)
        el = np.arcsin(np.divide(u, np.sqrt(np.square(e) + np.square(n) + np.square(u))))
        output = np.rad2deg(np.stack((azimuth, el), axis= 1)) % 360
        return output

    def calc_N_and_N1(self, phi):
        a = self.earth_semimajor
        b = self.earth_semiminor
        N = a ** 2 / np.sqrt(a ** 2 * (np.cos(phi) ** 2) + b ** 2 * (np.sin(phi) ** 2))
        N1 = (b / a) ** 2 * N
        return N, N1

    def rotation_matrix_around_z_then_x(self, longitude, latitude):
        theta = radians(longitude)
        phi= radians(latitude)
        R = np.empty([3,3])
        R[0, 0], R[0, 1], R[0, 2] = -sin(theta), cos(theta), 0
        R[1, 0], R[1, 1], R[1, 2] = -sin(phi) * cos(theta), -sin(theta) * sin(phi), cos(phi)
        R[2, 0], R[2, 1], R[2, 2] = cos(phi) * cos(theta), cos(phi) * sin(theta), sin(phi)
        return R

if __name__ == '__main__':
    transformer = CoordinateTransforms()
    boulder_geo = np.array([40 + 53/3600, -(105 + 16/60 + 13/3600), 1630])
    boulder_ecef = np.array([-1288648., -4720213.,4080224.])
    problem_4dot2_array = np.array([-2694685.473, -4293642.366, 3857878.924])
    longs_peak = np.array([40 + 15 / 60 + 18 / 3600, - (105 + 36 / 60 + 54 / 3600), 4346])
    longs_peak_ecef = transformer.geodetic_to_ecef(longs_peak)
    sat_ecef = np.array([[-16167171.36859017,  -1793649.91595709,  20921584.1223507],
               [-16935567.18921546,  -3748824.86903411,  20063787.30579629],
               [-17750745.18926655,  -5573022.91810005,  18931943.29873651],
               [-18588295.69129968,  -7243651.53318352,  17542129.88023571],
               [-19421271.95141234,  -8742805.41425279,  15913860.31182087],
               [-20221030.00060841, -10057677.79019538,  14069785.76311001],
               [-20958119.81758282, -11180814.24136535,  12035360.9571818 ],
               [-21603200.20988103, -12110203.8432249 ,   9838478.89827281],
               [-22127949.63927174, -12849207.39669298,   7509080.52069034]])
    # print("BOULDER ECEF COMP: ", "\n", boulder_ecef, "\n",  transformer.geodetic_to_ecef(boulder_geo))
    # print("BOULDER GEO COMP: ", "\n", boulder_geo, "\n", transformer.ecef_to_geodetic(boulder_ecef))
    # print(transformer.ecef_to_geodetic(problem_4dot2_array))
    # print("BOULDER LONGS PEAK ENU: ", transformer.ecef_to_enu(boulder_ecef, longs_peak_ecef))
    print(transformer.ecef2sky(boulder_ecef, sat_ecef))