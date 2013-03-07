import numpy as np
import cv2
import os
import logging
range = xrange

logging.basicConfig(level=logging.INFO)
# TODO !!!!! make sure these actually work as desired
# TODO write test cases 
def frame_rgb2yiq(frame):
    # frame is MxNx3
    # we want 3x(M*N)
    shape = frame.shape
    arrlen = shape[0]*shape[1]
    rgb_arr = np.reshape(frame, (arrlen, 3)).T
    yiq_arr = rgb2yiq(rgb_arr)
    yiq = np.reshape(yiq_arr.T, shape) # back into a frame
    return yiq

def rgb2yiq(rgb):
    # matrix * (R G B)'
    # take matrix from wikipedia
    # This matrix is sourced from MATLAB
    # http://www.mathworks.com/help/images/ref/rgb2ntsc.html
    # Retrieved 6 Mar 13
    convert = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]])
    yiq = np.dot(convert, rgb)
    return yiq

# TODO you probably need to fix domains and etc.
def yiq2rgb(yiq):
    # matrix * (Y I Q)'
    # take matrix from wikipedia
    convert = np.array([
        [1, 0.9563, 0.6210],
        [1, -0.2721, -0.6474],
        [1, -1.1070, 1.7046]])
    rgb = np.dot(convert, yiq)
    return rgb
    
# TODO you probably need to fix domains and etc.
def frame_yiq2rgb(frame):
    # frame is MxNx3
    # we want 3x(M*N)
    shape = frame.shape
    arrlen = shape[0]*shape[1]
    yiq_arr = np.reshape(frame, (arrlen, 3)).T
    rgb_arr = yiq2rgb(yiq_arr)
    rgb = np.reshape(rgb_arr.T, shape) # back into a frame
    return rgb

def iterate_pyramid(img, n):
    while n < 0:
        img = cv2.pyrDown(img)
        n += 1
    while n > 0:
        img = cv2.pyrUp(img)
        n -= 1
    return img

dataout = 'out'
fpath = 'EVM_Matlab/data/face.mp4'

os.mkdir(dataout)

vid = cv2.VideoCapture(fpath)
width = int(vid.get(3))
height = int(vid.get(4))
fps = vid.get(5)
length = int(vid.get(7)) # number of frames
length -= 10 # TODO check why they do this in paper/ code?

# 1. Create the pyramid stack
SCALE = 4

# Naming convention for frame variables is {if}{rgb,yiq}_frame
# i integer {0..255}; f float [0, 1); rgb or yiq color space
valid, ibgr_frame = vid.read()
import pdb;pdb.set_trace()
shape = iterate_pyramid(ibgr_frame, -SCALE).shape
logging.info('shape %s length %s', shape, length)
logging.info('Start allocating..')
stack = np.zeros(shape + (length,)) #create a 4-dim stack (x, y, color-channel, time index)
logging.info('Allocated stack. Reading in video...')

for i in range(length):
    irgb_frame = ibgr_frame[:,:,::-1]
    frgb_frame = irgb_frame.astype('float') / 255
    fyiq_frame = frame_rgb2yiq(frgb_frame)
    down3 = iterate_pyramid(fyiq_frame, -SCALE)
    stack[:,:,:,i] = down3
    valid, ibgr_frame = vid.read()
    # readframe is at the end because we've already done a read to get the pre-alloc size.
    # the last read will return null (or not null if we truncate the vidstream) and will 
    # be thrown away - exactly what we want

logging.info('Finished video read into stack. Performing bandpass filter...')
# 2. Perform the bandpass filter on it
sampling_rate = fps
low_freq = 5./6
hi_freq = 1.

freq = np.linspace(0.0, 1.0, length, endpoint=False) * sampling_rate # CHECK THIS
logging.info('Starting FFT of stack...')
F = np.fft.fft(stack) # F is complex-valued
logging.info('FFT of stack complete.')

mask = np.logical_and(low_freq < freq, freq < hi_freq)
F[:,:,:,mask] = 0
logging.info('FFT band-passed. Now compute the inverse...')
filtered_stack = np.real(np.fft.ifft(F))
logging.info('Inverse complete')

# 3. Do the attenuation
alpha = 50
chrom_atten = 1
filtered_stack[:,:,:,0] *= alpha
filtered_stack[:,:,:,1:3] *= alpha * chrom_atten

# 5. Write the attenuated signal out

vid.set(1, 0) # seek back to the first frame
logging.info('Begin writing out...')
vidout = cv2.VideoWriter(os.path.join(dataout, 'face.avi'), 
        cv2.cv.CV_FOURCC(*'DIVX'), int(fps), (width, height))
for i in range(length):
    valid, ibgr_frame = vid.read()
    irgb_frame = ibgr_frame[:,:,::-1]
    frgb_frame = irgb_frame.astype('float') / 255
    fyiq_frame = frame_rgb2yiq(frgb_frame)
    attenuated_full_frame = cv2.resize(filtered_stack[:,:,:,i], (width, height))
    result_frame = fyiq_frame + attenuated_full_frame
    result_frame[result_frame > 1] = 1
    result_frame[result_frame < 0] = 0
    result_frame = fyiq_frame
    frgb_resultframe = frame_yiq2rgb(result_frame)
    irgb_resultframe = (frgb_resultframe * 255).astype('uint8')
    ibgr_resultframe = irgb_resultframe[:,:,::-1]
    vidout.write(ibgr_resultframe)

logging.info('Analysis complete.')
