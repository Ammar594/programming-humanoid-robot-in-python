'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''
from pid import PIDAgent
from keyframes import rightBackToStand


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.startTime = -1

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}

        if self.startTime == -1:
            self.startTime = perception.time
        adjustTime = perception.time - self.startTime
        joints, times, keys = keyframes

        skipJoints = 0
        for (m, name) in enumerate(joints):
            minTime = 0
            maxTime = 0
            keyframe = 0
            jointTimes = times[m]

            if jointTimes[-1] < adjustTime:
                skipJoints += 1
                if skipJoints == len(joints):
                    self.startTime = -1
                    self.keyframes = ([], [], [])
                continue

            for n in range(len(jointTimes)):
                maxTime = jointTimes[n]

                if minTime <= adjustTime <= maxTime:
                    keyframe = n
                    break
                minTime = maxTime
            try:
                i = (adjustTime - minTime) / (maxTime - minTime)
            except:
                i = 0

            if keyframe == 0:
                p0 = 0
                p1 = 0
                p3 = keys[m][keyframe][0]
                p2 = p3 + keys[m][keyframe][1][2]

            else:
                p0 = keys[m][keyframe - 1][0]
                p1 = p0 + keys[m][keyframe - 1][2][2]
                p3 = keys[m][keyframe][0]
                p2 = p3 + keys[m][keyframe][1][2]

            angle = ((1 - i) ** 3) * p0 + 3 * i * ((1 - i) ** 2) * p1 + 3 * (i ** 2) * (1 - i) * p2 + (i ** 3) * p3

            target_joints[name] = angle
            if name == "LHipYawPitch":
                target_joints["RHipYawPitch"] = angle

        return target_joints



if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = rightBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
