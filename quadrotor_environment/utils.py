import random
from typing import Sequence

import numpy as np
import math
import quaternion


def rotate_vector(q: np.quaternion, vec: Sequence[float]):
    return (q.conjugate() * np.quaternion(0, vec[0], vec[1], vec[2]) * q).components[1:4]


# input: roll, pitch, yaw or np.array([roll, pitch, yaw])
# output: q
def euler2quat(*e):
    e = np.asarray(e[0]) if len(e) == 1 else np.asarray(e)
    cy = np.cos(e[2] * 0.5)
    sy = np.sin(e[2] * 0.5)
    cr = np.cos(e[0] * 0.5)
    sr = np.sin(e[0] * 0.5)
    cp = np.cos(e[1] * 0.5)
    sp = np.sin(e[1] * 0.5)

    q = np.zeros((4,))
    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[2] = cy * cr * sp + sy * sr * cp
    q[3] = sy * cr * cp - cy * sr * sp
    return q


def sample_random_rotation():
    #TODO: unclear if this one is completely correct but the bias shoud not be too bad
    alpha_len = 0
    while alpha_len == 0:
        alpha = np.random.randn(3) * [1, 1, 1]
        alpha_len = np.linalg.norm(alpha)
    alpha /= alpha_len
    amount = np.random.randn() * np.pi/2
    alpha *= amount
    return np.quaternion(0, *alpha).exp()


def sample_random_rotation2():
    return quaternion.from_euler_angles(np.random.uniform(0, np.pi * 2, 3))


def sample_random_rotation3():
    #TODO: this one is the one used by HWANGBO and it looks a lot nicer. But should it look nice?
    v = np.random.rand(4) * 2 - 1
    v /= np.linalg.norm(v)
    return np.quaternion(*v)

# @jit
def inverted_huber_loss(x, d=2):  # TODO: rename this one
    if x > 0:
        low_range_val = np.sqrt(2 * x)
        return low_range_val if low_range_val < d else x/d + 0.5 * d
    else:
        low_range_val = np.sqrt(2 * -x)
        return -low_range_val if low_range_val < d else x/d - 0.5 * d

def quat2eul(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    ysqr = y ** 2

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array((roll, pitch, yaw))

def quat2eulzyx(q):
    q0, q1, q2, q3 = q.components

    v1_x = 2 * (q0 * q0 + q1 * q1) - 1
    v1_y = 2 * (q0 * -q3 + q1 * q2)

    v2_x = 2 * (q0 * q2 + q3 * q1)
    v2_y = 2 * (-q0 * q1 + q3 * q2)
    v2_z = 2 * (q0 * q0 + q3 * q3) - 1

    if v2_x < -0.9999999999 or v2_x > 0.9999999999:
        return np.zeros(3)

    return np.array((math.atan2(v1_y, v1_x), -math.asin(v2_x), math.atan2(v2_y, v2_z)))

def eulNASA2rotmat(eulNASA):

    heading, attitude, bank = eulNASA
    c1 = np.cos(heading)
    s1 = np.sin(heading)
    c2 = np.cos(attitude)
    s2 = np.sin(attitude)
    c3 = np.cos(bank)
    s3 = np.sin(bank)

    m00 = c1 * c2
    m01 = -s1 * c2
    m02 = s2
    m10 = s1 * c3 + (c1 * s2 * s3)
    m11 = (c1 * c3) - (s1 * s2 * s3)
    m12 = -c2 * s3
    m20 = (s1 * s3) - (c1 * s2 * c3)
    m21 = (c1 * s3) + (s1 * s2 * c3)
    m22 = c2 * c3
    return np.array([m00, m01, m02, m10, m11, m12, m20, m21, m22])

if __name__ == '__main__':
    with open("sampled_rotations.dat", "w") as fh:
        for _ in range(2000):
            rot = "\t".join(str(e) for e in sample_random_rotation3().passiveRotateVec([0, 0, 1]))
            fh.write(rot + "\n")



