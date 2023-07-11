import numpy as np
import soundfile as sf

class Drc:
    def getInfo():
        return {
            'category': 'COMMON',
            'name': 'Drc',
            'text': 'Drc',
            'inBufIdxs': '',
            'outBufIdxs': '',
            'fields': [
                ['numChanIn', 'No. of Inputs', '2', 'text', ''],
                ['numChanOut', 'No. of Outputs', '2', 'text', ''],
                ['inBufIdxs', 'Input Index', '', 'text', 'disabled'],
                ['outBufIdxs', 'Output Index', '', 'text', 'disabled'],
                ['isGainBuf', 'Gain Buffer', 'Yes', 'text', 'disabled'],
                ['drcType', 'DRC Type', 'NORMAL', 'select', 'drcType'],
                ['blockLen', 'Block Length', '64', 'text', ''],
                ['afterGainDb', 'AfterGain (dB)', '3', 'text', ''],
                ['thresholdDb', 'Threshold (dB)', '-3', 'text', ''],
                ['ratio', 'Ratio', '5', 'text', ''],
                ['width', 'Softknee Width', '5', 'text', ''],
                ['attackMs', 'Attack (ms)', '50', 'text', ''],
                ['holdMs', 'Hold (ms)', '200', 'text', ''],
                ['releaseMs', 'Release (ms)', '100', 'text', ''],
                ['lvlDetector', 'Level Detector', 'PEAK', 'select', 'lvlDetector'],
                ['debugFlag', 'Debug Flag', 'NO', 'select', 'yesNo'],
            ]
        }

    def __init__(self, fields):
        params = {}
        for field in fields:
            params[field[0]] = field[2]
        self.key = int(np.abs(params.get('key')))
        self.name = Drc.getInfo().get('name')
        self.debugFlag = params.get('debugFlag') == 'YES'
        self.inBufIdxs = np.array([int(i) for i in params.get('inBufIdxs').split(',')])
        self.outBufIdxs = np.array([int(i) for i in params.get('outBufIdxs').split(',')])
        self.numChanIn = len(params.get('inBufIdxs').split(','))
        self.numChanOut = len(params.get('outBufIdxs').split(','))
        self.blockLen = np.int16(params.get('blockLen'))
        self.thresholdDb = float(params.get('thresholdDb'))
        self.threshold = 10**(self.thresholdDb/20)
        self.afterGainDb = float(params.get('afterGainDb'))
        self.afterGain = 10**(self.afterGainDb/20)
        self.ratio = float(params.get('ratio'))
        self.width = max(float(params.get('width')), 0.1)
        self.softkneeBgnDb = self.thresholdDb - self.width/2
        self.softkneeEndDb = self.thresholdDb + self.width/2
        self.softkneeBgn = 10**(self.softkneeBgnDb/20)
        self.softkneeEnd = 10**(self.softkneeEndDb/20)
        if params.get('drcType') == 'NORMAL':
            self.attack = int(np.round(float(params['attackMs'])/1000*48000/self.blockLen))
            self.hold = int(np.round(float(params['holdMs'])/1000*48000/self.blockLen))
            self.release = int(np.round(float(params['releaseMs'])/1000*48000/self.blockLen))
            self.currMax = 0
            self.max = 0
            self.cGain = 1
            self.tGain = 1
            self.counter = 0
            self.state = 'NONE'
            self.arrayArange = np.arange(self.blockLen)
            self.arrayOne = np.ones(self.blockLen)
            self.step = self.stepNormal
        elif params.get('drcType') == 'HISTEN':
            self.attack = np.exp(-1 / (max(float(params['attackMs']), 0)/3000*48000))
            self.release = np.exp(-1 / (max(float(params['releaseMs']), 0)/3000*48000))
            self.ratioInv = 1/self.ratio
            self.yl = 0
            self.yL = 0
            self.left = self.blockLen % 3
            self.step = self.stepHisten
        elif params.get('drcType') == 'SIMPLE':
            maxgainDb = -self.thresholdDb + (self.thresholdDb / self.ratio)
            self.attackGainDb = maxgainDb * (self.blockLen / (float(params['attackMs']) * 0.001 * 48000))
            self.decayGainDb = maxgainDb * (self.blockLen / (float(params['releaseMs']) * 0.001 * 48000))
            self.cGain1 = 1.0
            self.cGain2 = 1.0
            self.destGainDb = 0
            self.energySmoothed = 0
            self.step = self.stepSimple
        self.computeLvl = self.computePeakLvl if params.get('lvlDetector')=='PEAK' else self.computeRmsLvl
        if self.debugFlag:
            self.fname = r'./modules/debug/debug_{}{}.pcm'.format(self.key, self.name)
            with open(self.fname, 'w') as f:
                pass
        
    def setFrameLen(self, frameLen):
        self.frameLen = frameLen
        self.Nb = self.frameLen // self.blockLen

    def apply(self, gBuffer):
        i = 0
        for _ in range(self.Nb):
            x = gBuffer[i:i+self.blockLen, self.inBufIdxs]
            y = self.step(x)
            gBuffer[i:i+self.blockLen, self.outBufIdxs] = y[:, None]
            i += self.blockLen
        if self.debugFlag:
            with open(self.fname, 'a') as f:
                (np.array(gBuffer[:, self.outBufIdxs]).astype(float).reshape(-1)*32767).astype(np.int16).tofile(f)
        return gBuffer
    
    def computePeakLvl(self, x):
        maxX = np.abs(x).max()
        return maxX, 20*np.log10(max(maxX, 1e-10))
    
    def computeRmsLvl(self, x):
        maxX = np.sqrt(np.mean(x**2, axis=0)).max()
        return maxX, 20*np.log10(max(maxX, 1e-10))
    
    def computeTargetGain(self, x, isDb=False):
        if not isDb:
            xDb = 20*np.log10(x)
        else:
            xDb = x
            x = 10**(xDb/20)
        overshoot = xDb - self.thresholdDb
        if xDb > self.softkneeEndDb:
            yDb = self.thresholdDb + overshoot/self.ratio
        elif xDb < self.softkneeBgnDb:
            yDb = self.thresholdDb
        else:
            yDb = xDb + (1/self.ratio-1)*(overshoot-self.thresholdDb+self.width/2)**2/(2*self.width)
        y = 10**(yDb/20)
        return y/x, yDb-xDb

    def stepHisten(self, x):
        dataLen = x.shape[0]
        dataDsLen = dataLen//3 
        gainDb = np.zeros(dataDsLen)
        
        #  downsample and obtain the highest energy points
        tmp = np.max(np.abs(x[:dataLen, :]), axis=1)

        tmp = tmp[::3]  # envelope 
 
        # db level
        tmp = 20*np.log10(np.fmax(tmp, 1e-10))

        # slow signal energy estimate
        for i in range(dataDsLen):
            overshoot = tmp[i] - self.thresholdDb
            if overshoot <= -0.5 * self.width:
                gainDb[i] = tmp[i]
            elif overshoot > 0.5 * self.width:
                gainDb[i] = self.thresholdDb + overshoot * self.ratioInv
            else:
                gainDb[i] = tmp[i] + 0.5/self.width*(self.ratioInv-1.0)*(overshoot+0.5*self.width)**2
            gainDb[i] = tmp[i] - gainDb[i]

            if gainDb[i] > self.yL:                                          # branch
                self.yL = self.attack*self.yL + (1-self.attack)*gainDb[i]
            else:
                self.yL = self.release*self.yL + (1-self.release)*gainDb[i]

            # self.yl = max(gainDb[i], self.release*self.yl+(1-self.release)*gainDb[i])   # decouple
            # self.yL = self.attack*self.yL + (1-self.attack)*self.yl

            gainDb[i] = self.yL
        
        gainDb = self.afterGainDb - gainDb
        lg = 10**(gainDb/20)

        # upsample single channel
        gain = np.zeros(dataLen)
        gain[::3] = lg
        for i in range(lg.shape[0]-1):
            gain[1+3*i] = (2.0*gain[3*i] + gain[3+3*i]) * 0.333333343
            gain[2+3*i] = (gain[3*i] + 2.0*gain[3+3*i]) * 0.333333343
        extra = lg[-1] + (lg[-1] - lg[-2])
        if self.left == 1:
            gain[-1] = (2.0*gain[-3] + extra) * 0.333333343
        else:
            gain[-2] = (2.0*gain[-3] + extra) * 0.333333343
            gain[-1] = (gain[-3] + 2.0*extra) * 0.333333343

        return gain
        
    # Implement a simple compressor
    # The code here is highly inefficient but easy to read
    # To speed it up all possible log- and power functions must be
    # precomputed in the service or init functions.
    # On top the remaining log/pow should be replaced by a fast approximation
    def stepSimple(self, x):
        # Also here working in-place on audiobuf is not possible,
        # so we create a new array which we return - aarrggh
        gain = np.zeros(self.blockLen, dtype=np.float32)

        # Compute the energy of the block
        # To extend the compressor to stereo add up the energy of both
        # channels here and half the result. Then calculate everything
        # as if it would be one channel and apply the resulting gain
        # to both channels
        energy = np.mean(x**2)

        # Smooth the energy curve, maybe do this faster for increases in energy
        # self.energy_smoothed = 0.98*self.energy_smoothed + 0.02*energy/blocksize
        self.energySmoothed = 0.95 * self.energySmoothed + 0.05 * energy

        # Calculat the dB value for the averaged energy
        energyDb = 10 * np.log10(self.energySmoothed)

        # Compute the target gain
#         if (energy_db < self.threshold_db):
#             # The current energy is below the threshold, so go for the maximum boosting
#             target_gain_db = self.aftergain_db
#         else:
#             # We are in compression range, so reduce the aftergain accordingly
#             target_gain_db = self.aftergain_db - \
#                              (energy_db - self.threshold_db) * (1.0 - 1.0 / self.ratio)
        _, targetGainDb = self.computeTargetGain(energyDb, True)
        targetGainDb = targetGainDb + energyDb + self.afterGainDb

        # Follow the target gain but limit the slope to the attack/decay settings
        # And use fmin/fmax to avoid overshooting the target which would cause oscillation
        if (targetGainDb > self.destGainDb):
            self.destGainDb = min((self.destGainDb + self.decayGainDb), targetGainDb)
        else:
            self.destGainDb = max((self.destGainDb - self.attackGainDb), targetGainDb)

        # Now apply the
        destGain = np.float32(np.power(2.0, self.destGainDb / 6))
        cGain1 = self.cGain1
        cGain2 = self.cGain2
        for i in range(self.blockLen):
            #            curr_gain_lin = 0.998*curr_gain_lin + 0.002*dest_gain_lin
            # Use a 12dB filter, 1 or 2 real biquads with 24dB slope should even be better
            cGain1 = 0.98 * cGain1 + 0.02 * destGain
            cGain2 = 0.98 * cGain2 + 0.02 * cGain1
            gain[i] = cGain2

        self.cGain1 = cGain1
        self.cGain2 = cGain2

        return gain

    def stepNormal(self, x):
        currMax, _ = self.computeLvl(x)
        self.currMax = 0.95*self.currMax + 0.05*currMax
        
        if self.currMax > self.softkneeBgn:
            if self.currMax > self.max:
                self.state = 'ATTACK'
                self.max = self.currMax
                self.counter = self.attack
                self.tGain, _ = self.computeTargetGain(self.max, False)
                self.dGain = (self.tGain - self.cGain)/(self.blockLen*self.attack)
                gainBuffer = self.cGain + self.dGain*self.arrayArange
                self.cGain = gainBuffer[-1]
            else:
                if self.state == 'ATTACK':
                    gainBuffer = self.cGain + self.dGain*self.arrayArange
                    self.cGain = gainBuffer[-1]
                elif self.state == 'HOLD':
                    gainBuffer = self.arrayOne * self.cGain
                else:
                    self.state = 'ATTACK'
                    self.counter = self.attack
                    self.tGain, _ = self.computeTargetGain(self.max, False)
                    diff = self.tGain - self.cGain
                    attack = self.attack if diff < 0 else self.hold
                    self.dGain = diff/(self.blockLen*attack)
                    gainBuffer = self.cGain + self.dGain*self.arrayArange
                    self.cGain = gainBuffer[-1]
        else:
            if currMax > self.max:
                self.max = currMax
            if self.state == 'ATTACK' or self.state == 'RELEASE':
                gainBuffer = self.cGain + self.dGain*self.arrayArange
                self.cGain = gainBuffer[-1]
            elif self.state == 'HOLD' or self.state == 'NONE':
                gainBuffer = self.arrayOne * self.cGain
        
        if self.state != 'ATTACK':
            self.max *= 0.9998
            if self.max < self.threshold:
                self.max = self.threshold
        self.counter -= 1
        if self.state == 'NONE' or self.counter > 0:
            return gainBuffer*self.afterGain
        
        if self.state == 'ATTACK':
            self.state = 'HOLD'
            self.dGain = 0
            self.cGain = self.tGain
            self.counter = self.hold
        elif self.state == 'HOLD':
            self.state = 'RELEASE'
            self.tGain = 1
            self.dGain = (self.tGain - self.cGain)/(self.blockLen*self.release)
            self.counter = self.release
        elif self.state == 'RELEASE':
            self.state = 'NONE'
            self.tGain = 1
            self.cGain = 1
            self.dGain = 0
        return gainBuffer*self.afterGain

# if __name__ == '__main__':
#     frameLen = 480
#     blockLen = 480
#     attackMs = 100
#     releaseMs = 1000
#     attack = int(np.round(attackMs/1000*48000))
#     thresholdDb = -20
#     drcType = 'HISTEN'
#     fields = [
#         ['key', 'Key', 1, 'text', ''],
#         ['numChanIn', 'No. of Inputs', '2', 'text', ''],
#         ['numChanOut', 'No. of Outputs', '1', 'text', ''],
#         ['inBufIdxs', 'Input Index', '0,1', 'text', 'disabled'],
#         ['outBufIdxs', 'Output Index', '2', 'text', 'disabled'],
#         ['drcType', 'Drc Type', drcType, 'select', 'drcType'],
#         ['blockLen', 'Block Length', blockLen, 'text', ''],
#         ['afterGainDb', 'AfterGain (dB)', '5', 'text', ''],
#         ['thresholdDb', 'Threshold (dB)', '-20', 'text', ''],
#         ['ratio', 'Ratio', '10', 'text', ''],
#         ['width', 'Softknee Width', '5', 'text', ''],
#         ['attackMs', 'Attack (ms)', attackMs, 'text', ''],
#         ['holdMs', 'Hold (ms)', '200', 'text', ''],
#         ['releaseMs', 'Release (ms)', releaseMs, 'text', ''],
#         ['lvlDetector', 'Level Detector', 'PEAK', 'select', 'lvlDetector'],
#         ['debugFlag', 'Debug Flag', 'NO', 'select', 'yesNo'],
#     ]
#     drc = Drc(fields)
#     drc.setFrameLen(frameLen)

#     x, fs = sf.read(r'D:/Histen/Sounds/Input/test.wav')
#     # x = x[:30*fs, :]
#     gBuffer = np.zeros([frameLen, 6])
#     aBuffer = np.zeros([x.shape[0], 6])

#     currI = 0
#     while currI + frameLen < x.shape[0]:
#         gBuffer[:, [0,1]] = x[currI:currI+frameLen, :]
#         drc.apply(gBuffer)
#         aBuffer[currI:currI+frameLen, :] = gBuffer
#         currI += frameLen
#     # final_sig[:,10:12] = aBuffer[:,:2]

#     # if drcType == 'NORMAL':
#     #     delay = attack+blockLen
#     # elif drcType == 'SIMPLE':
#     #     delay = 0
#     # else:
#     #     delay = blockLen
#     # res = aBuffer[:, 2][:,None]*np.roll(aBuffer[:,:2], delay, axis=0)
#     # aBuffer[:, 0:2] = res
#     # aBuffer[:, 3:5] = x 

#     sf.write(r'D:/Histen/Sounds/Input/test_histen111.wav'.format(drcType), aBuffer[:,:2], fs)
