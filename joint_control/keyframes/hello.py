

def hello():
    # Choregraphe bezier export in Python.
    names = list()
    times = list()
    keys = list()

    names.append("HeadPitch")
    times.append([0.80000, 1.56000, 2.24000, 2.80000, 3.48000, 4.60000])
    keys.append([[0.29602, [3, -0.26667, 0.00000], [3, 0.25333, 0.00000]], [-0.17032, [3, -0.25333, 0.11200], [3, 0.22667, -0.10021]], [-0.34059, [3, -0.22667, 0.00000], [3, 0.18667, 0.00000]], [-0.05987, [3, -0.18667, 0.00000], [3, 0.22667, 0.00000]], [-0.19333, [3, -0.22667, 0.00000], [3, 0.37333, 0.00000]], [-0.01078, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("HeadYaw")
    times.append([0.80000, 1.56000, 2.24000, 2.80000, 3.48000, 4.60000])
    keys.append([[-0.13503, [3, -0.26667, 0.00000], [3, 0.25333, 0.00000]], [-0.35133, [3, -0.25333, 0.04939], [3, 0.22667, -0.04419]], [-0.41576, [3, -0.22667, 0.00372], [3, 0.18667, -0.00307]], [-0.41882, [3, -0.18667, 0.00307], [3, 0.22667, -0.00372]], [-0.52007, [3, -0.22667, 0.00000], [3, 0.37333, 0.00000]], [-0.37587, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("LElbowRoll")
    times.append([0.72000, 1.48000, 2.16000, 2.72000, 3.40000, 4.52000])
    keys.append([[-1.37902, [3, -0.24000, 0.00000], [3, 0.25333, 0.00000]], [-1.29005, [3, -0.25333, -0.03454], [3, 0.22667, 0.03091]], [-1.18267, [3, -0.22667, 0.00000], [3, 0.18667, 0.00000]], [-1.24863, [3, -0.18667, 0.02055], [3, 0.22667, -0.02496]], [-1.31920, [3, -0.22667, 0.00000], [3, 0.37333, 0.00000]], [-1.18421, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("LElbowYaw")
    times.append([0.72000, 1.48000, 2.16000, 2.72000, 3.40000, 4.52000])
    keys.append([[-0.80386, [3, -0.24000, 0.00000], [3, 0.25333, 0.00000]], [-0.69188, [3, -0.25333, -0.01372], [3, 0.22667, 0.01227]], [-0.67960, [3, -0.22667, -0.01227], [3, 0.18667, 0.01011]], [-0.61057, [3, -0.18667, 0.00000], [3, 0.22667, 0.00000]], [-0.75324, [3, -0.22667, 0.00000], [3, 0.37333, 0.00000]], [-0.67040, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

   # names.append("LHand")
   # times.append([1.48000, 4.52000])
  #  keys.append([[0.00416, [3, -0.49333, 0.00000], [3, 1.01333, 0.00000]], [0.00419, [3, -1.01333, #0.00000], [3, 0.00000, 0.00000]]])

    names.append("LShoulderPitch")
    times.append([0.72000, 1.48000, 2.16000, 2.72000, 3.40000, 4.52000])
    keys.append([[1.11824, [3, -0.24000, 0.00000], [3, 0.25333, 0.00000]], [0.92803, [3, -0.25333, 0.00000], [3, 0.22667, 0.00000]], [0.94030, [3, -0.22667, 0.00000], [3, 0.18667, 0.00000]], [0.86207, [3, -0.18667, 0.00000], [3, 0.22667, 0.00000]], [0.89735, [3, -0.22667, 0.00000], [3, 0.37333, 0.00000]], [0.84212, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("LShoulderRoll")
    times.append([0.72000, 1.48000, 2.16000, 2.72000, 3.40000, 4.52000])
    keys.append([[0.36352, [3, -0.24000, 0.00000], [3, 0.25333, 0.00000]], [0.22699, [3, -0.25333, 0.02572], [3, 0.22667, -0.02301]], [0.20398, [3, -0.22667, 0.00000], [3, 0.18667, 0.00000]], [0.21779, [3, -0.18667, -0.00670], [3, 0.22667, 0.00813]], [0.24847, [3, -0.22667, 0.00000], [3, 0.37333, 0.00000]], [0.22699, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

   # names.append("LWristYaw")
    #times.append([1.48000, 4.52000])
   # keys.append([[0.14722, [3, -0.49333, 0.00000], [3, 1.01333, 0.00000]], [0.11961, [3, -1.01333, #0.00000], [3, 0.00000, 0.00000]]])

    names.append("RElbowRoll")
    times.append([0.64000, 1.40000, 1.68000, 2.08000, 2.40000, 2.64000, 3.04000, 3.32000, 3.72000, 4.44000])
    keys.append([[1.38524, [3, -0.21333, 0.00000], [3, 0.25333, 0.00000]], [0.24241, [3, -0.25333, 0.00000], [3, 0.09333, 0.00000]], [0.34907, [3, -0.09333, -0.09496], [3, 0.13333, 0.13565]], [0.93425, [3, -0.13333, 0.00000], [3, 0.10667, 0.00000]], [0.68068, [3, -0.10667, 0.14138], [3, 0.08000, -0.10604]], [0.19199, [3, -0.08000, 0.00000], [3, 0.13333, 0.00000]], [0.26180, [3, -0.13333, -0.06981], [3, 0.09333, 0.04887]], [0.70722, [3, -0.09333, -0.10397], [3, 0.13333, 0.14852]], [1.01927, [3, -0.13333, -0.06647], [3, 0.24000, 0.11965]], [1.26559, [3, -0.24000, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("RElbowYaw")
    times.append([0.64000, 1.40000, 2.08000, 2.64000, 3.32000, 3.72000, 4.44000])
    keys.append([[-0.31298, [3, -0.21333, 0.00000], [3, 0.25333, 0.00000]], [0.56447, [3, -0.25333, 0.00000], [3, 0.22667, 0.00000]], [0.39113, [3, -0.22667, 0.03954], [3, 0.18667, -0.03256]], [0.34818, [3, -0.18667, 0.00000], [3, 0.22667, 0.00000]], [0.38192, [3, -0.22667, -0.03375], [3, 0.13333, 0.01985]], [0.97738, [3, -0.13333, 0.00000], [3, 0.24000, 0.00000]], [0.82678, [3, -0.24000, 0.00000], [3, 0.00000, 0.00000]]])

   # names.append("RHand")
   # times.append([1.40000, 3.32000, 4.44000])
   # keys.append([[0.01490, [3, -0.46667, 0.00000], [3, 0.64000, 0.00000]], [0.01492, [3, -0.64000, #0.00000], [3, 0.37333, 0.00000]], [0.00742, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("RShoulderPitch")
    times.append([0.64000, 1.40000, 2.08000, 2.64000, 3.32000, 4.44000])
    keys.append([[0.24702, [3, -0.21333, 0.00000], [3, 0.25333, 0.00000]], [-1.17193, [3, -0.25333, 0.00000], [3, 0.22667, 0.00000]], [-1.08910, [3, -0.22667, 0.00000], [3, 0.18667, 0.00000]], [-1.26091, [3, -0.18667, 0.00000], [3, 0.22667, 0.00000]], [-1.14892, [3, -0.22667, -0.11198], [3, 0.37333, 0.18444]], [1.02015, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    names.append("RShoulderRoll")
    times.append([0.64000, 1.40000, 2.08000, 2.64000, 3.32000, 4.44000])
    keys.append([[-0.24241, [3, -0.21333, 0.00000], [3, 0.25333, 0.00000]], [-0.95419, [3, -0.25333, 0.00000], [3, 0.22667, 0.00000]], [-0.46024, [3, -0.22667, 0.00000], [3, 0.18667, 0.00000]], [-0.96033, [3, -0.18667, 0.00000], [3, 0.22667, 0.00000]], [-0.32832, [3, -0.22667, -0.04750], [3, 0.37333, 0.07823]], [-0.25008, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    #names.append("RWristYaw")
   # times.append([1.40000, 3.32000, 4.44000])
   # keys.append([[-0.31298, [3, -0.46667, 0.00000], [3, 0.64000, 0.00000]], [-0.30377, [3, -0.64000, #-0.00920], [3, 0.37333, 0.00537]], [0.18250, [3, -0.37333, 0.00000], [3, 0.00000, 0.00000]]])

    return names, times, keys
