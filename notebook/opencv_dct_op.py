import cv2
import numpy as np

QY=np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,48,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

QC=np.array([[17,18,24,47,99,99,99,99],
                         [18,21,26,66,99,99,99,99],
                         [24,26,56,99,99,99,99,99],
                         [47,66,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99]])

def rgb2bgr(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, 

def bgr2rbg(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def rgb2ycrcb(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return image

def bgr2ycrcb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image

def block_cropping(image, blocksize=8):
    print('orginal image shape:', image.shape)
    h,w=(np.array(image.shape[:2])//blocksize * blocksize).astype(int)
    image=image[:h,:w]
    print('modified image shape:', image.shape)
    return image

def subsample_chrominance(YCbCr_image, SSV=2, SSH=2):
    crf=cv2.boxFilter(YCbCr_image[:,:,1], ddepth=-1, ksize=(2,2))
    cbf=cv2.boxFilter(YCbCr_image[:,:,2], ddepth=-1, ksize=(2,2))
    print(type(crf[0,0]),YCbCr_image[:,:,1].shape, crf.shape, cbf.shape)
    print("the difference of before & after Filter:", np.sqrt(np.sum(YCbCr_image[:,:,1]-crf) ** 2))
    crsub=crf[::SSV,::SSH] # sample with stride SSV in row and stride SSH in col.
    cbsub=cbf[::SSV,::SSH]
    print(crsub.shape, cbsub.shape)
    imSub_list=[YCbCr_image[:,:,0],crsub,cbsub]
    return imSub_list

def quality_factorize(qy, qc, QF=99):
    if QF < 50 and QF > 1:
        scale = np.floor(5000/QF)
    elif QF < 100:
        scale = 200-2*QF
    else:
        print("Quality Factor must be in the range [1..99]")
    scale=scale/100.0
    print("Q factor:{}, Q scale:{} ".format(QF, scale))
    q=[qy*scale,qc*scale,qc*scale]
    return q

def DCT_encoder(imSub_list, q, blocksize=8, thresh = 0.05):
    TransAll_list=[]
    TransAllThresh_list=[]
    TransAllQuant_list=[]
    for idx,channel in enumerate(imSub_list):
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        Trans = np.zeros((channelrows,channelcols), np.float32)
        TransThresh = np.zeros((channelrows,channelcols), np.float32)
        TransQuant = np.zeros((channelrows,channelcols), np.float32)
        blocksV=int(channelrows/blocksize)
        blocksH=int(channelcols/blocksize)
        vis0 = np.zeros((channelrows,channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis0=vis0-128 # before DCT the pixel values of all channels are shifted by -128
        for row in range(blocksV):
            for col in range(blocksH):
                currentblock = cv2.dct(vis0[row*blocksize:(row+1)*blocksize,col*blocksize:(col+1)*blocksize])
                Trans[row*blocksize:(row+1)*blocksize,col*blocksize:(col+1)*blocksize]=currentblock
                thres_block=TransThresh[row*blocksize:(row+1)*blocksize,col*blocksize:(col+1)*blocksize]=currentblock\
                                                            * (abs(currentblock) > thresh*np.max(currentblock))
                TransQuant[row*blocksize:(row+1)*blocksize,col*blocksize:(col+1)*blocksize]=np.round(thres_block/q[idx])
        TransAll_list.append(Trans)
        TransAllThresh_list.append(TransThresh)
        TransAllQuant_list.append(TransQuant)
    return TransAll_list, TransAllThresh_list ,TransAllQuant_list

def IDCT_decoder(TransAllQuant_list, q, blocksize=8):
    h, w = TransAllQuant_list[0].shape
    c = len(TransAllQuant_list)
    DecAll=np.zeros((h,w,c), np.uint8)
    for idx,channel in enumerate(TransAllQuant_list):
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        blocksV=int(channelrows/blocksize)
        blocksH=int(channelcols/blocksize)
        back0 = np.zeros((channelrows,channelcols), np.uint8)
        for row in range(blocksV):
            for col in range(blocksH):
                dequantblock=channel[row*blocksize:(row+1)*blocksize,col*blocksize:(col+1)*blocksize]*q[idx]
                currentblock = np.round(cv2.idct(dequantblock))+128 # inverse shiftign of the shift of the pixel values, sucht that their value range is [0,...,255].
                currentblock[currentblock>255]=255
                currentblock[currentblock<0]=0
                back0[row*blocksize:(row+1)*blocksize,col*blocksize:(col+1)*blocksize]=currentblock
        print('back0 shape:', back0.shape)
        back1=cv2.resize(back0,(w,h)) # the subsampled chrominance channels are interpolated, using cv2.INTER_LINEAR in default. 
        print('back1 shape:', back1.shape)
        DecAll[:,:,idx]=np.round(back1)
    return DecAll

def mse(image1, image2):
    SSE=np.sqrt(np.sum((image1-image2)**2))
    return SSE
