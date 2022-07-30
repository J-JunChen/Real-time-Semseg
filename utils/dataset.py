import os
import os.path 
import cv2
import numpy as np

from torch.utils.data import Dataset
import cityscapesscripts.helpers.labels as CSLabels
import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval
from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError('Image list file do not exist:' + data_list + '\n'))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', img_type='rgb', data_root=None, data_list=None, transform=None):
        self.split = split
        self.img_type = img_type
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR 3 channel ndarray with shape H * W * 3
        if self.img_type == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert cv2 read image from BGR order to RGB order
        elif self.img_type == 'dct':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # convert cv2 read image from BGR to YUV color space
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image = np.concatenate((image_rgb, image_yuv), axis=2)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label


class SemDataDCT(Dataset):
    def __init__(self, 
                 split='train', 
                 data_root=None, 
                 data_list=None, 
                 block_size=8, 
                 sub_sampling='4:2:0', 
                 quality_factor=99, 
                 threshold=0.0, 
                 transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.block_size = block_size
        self.sub_sampling = sub_sampling
        self.quality_factor = quality_factor
        self.thresh = threshold
        self.transform = transform
        # the quantisation matrices for the luminace channel (QY)
        self.QY=np.array([[16,11,10,16,24,40,51,61],
                                [12,12,14,19,26,48,60,55],
                                [14,13,16,24,40,57,69,56],
                                [14,17,22,29,51,87,80,62],
                                [18,22,37,56,68,109,103,77],
                                [24,35,55,64,81,104,113,92],
                                [49,64,78,87,103,121,120,101],
                                [72,92,95,98,112,100,103,99]])
        # the quantisation matrices for the chrominance channels (QC)
        self.QC=np.array([[17,18,24,47,99,99,99,99],
                                [18,21,26,66,99,99,99,99],
                                [24,26,56,99,99,99,99,99],
                                [47,66,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99],
                                [99,99,99,99,99,99,99,99]])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR 3 channel ndarray with shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # convert cv2 read image from BGR  to YUV color space
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        # 1. Ensure the size of image and label satisfy the factor of 8.
        image = self.block_cropping(image, self.block_size)
        label = self.block_cropping(label, self.block_size)
        # 2. The chrominance channels Cr and Cb are subsampled
        if self.sub_sampling == '4:2:0':
            imSub = self.subsample_chrominance(image, 2, 2)
        # 3. Get the quatisation matrices, which will be applied to the DCT coefficients
        Q = self.quality_factorize(self.QY, self.QC, self.quality_factor)
        # 4. Apply DCT algorithm for orignal image
        TransAll, TransAllThresh ,TransAllQuant = self.dct_encoder(imSub, Q, self.block_size, self.thresh)
        # 5. Split the same frequency in each 8x8 blocks to the same channel
        dct_list = self.split_frequency(TransAll, self.block_size)
        # 6. upsample the Cr & Cb channel to concatenate with Y channel
        dct_coefficients = self.upsample(dct_list)
        if self.transform is not None:
            dct_coefficients, label = self.transform(dct_coefficients, label)
        return dct_coefficients, label
    
    def block_cropping(self, image, blocksize=8):
        B = blocksize
        # print('orginal image shape:', image.shape)
        h, w = (np.array(image.shape[:2]) // B * B).astype(int)
        image = image[:h, :w]
        # print('modified image shape:', image.shape)
        return image
    
    def subsample_chrominance(self, YCbCr_image, SSV=2, SSH=2):
        crf = cv2.boxFilter(YCbCr_image[:,:,1], ddepth=-1, ksize=(2,2))
        cbf = cv2.boxFilter(YCbCr_image[:,:,2], ddepth=-1, ksize=(2,2))
        crsub = crf[::SSV, ::SSH] # sample with stride SSV in row and stride SSH in col.
        cbsub = cbf[::SSV, ::SSH]
        imSub_list = [YCbCr_image[:, :, 0], crsub, cbsub]
        return imSub_list

    def quality_factorize(self, qy, qc, QF=99):
        if QF < 50 and QF > 1:
            scale = np.floor(5000/QF)
        elif QF < 100:
            scale = 200-2*QF
        else:
            print("Quality Factor must be in the range [1..99]")
        scale = scale / 100.0
        # print("Q factor:{}, Q scale:{} ".format(QF, scale))
        q = [qy*scale, qc*scale, qc*scale]
        return q
    
    def dct_encoder(self, imSub_list, Q, blocksize=8, thresh=0.05):
        TransAll_list = []
        TransAllThresh_list = []
        TransAllQuant_list = []
        B = blocksize
        for idx, channel in enumerate(imSub_list):
            channelrows = channel.shape[0]
            channelcols = channel.shape[1]
            Trans = np.zeros((channelrows, channelcols), np.float32)
            TransThresh = np.zeros((channelrows, channelcols), np.float32)
            TransQuant = np.zeros((channelrows, channelcols), np.float32)
            blocksV = int(channelrows / B)
            blocksH = int(channelcols / B)
            vis0 = np.zeros((channelrows,channelcols), np.float32)
            vis0[:channelrows, :channelcols] = channel
            vis0 = vis0-128 # before DCT the pixel values of all channels are shifted by -128
            for row in range(blocksV):
                for col in range(blocksH):
                    currentblock = cv2.dct(vis0[row*B:(row+1)*B, col*B:(col+1)*B])
                    Trans[row*B:(row+1)*B, col*B:(col+1)*B] = currentblock
                    thres_block = TransThresh[row*B:(row+1)*B, col*B:(col+1)*B] = \
                        currentblock * (abs(currentblock) > thresh * np.max(currentblock))
                    TransQuant[row*B:(row+1)*B, col*B:(col+1)*B] = np.round(thres_block / Q[idx])
            TransAll_list.append(Trans)
            TransAllThresh_list.append(TransThresh)
            TransAllQuant_list.append(TransQuant)
            # print('If TransAll_List is the same as TransAllTresh_List? ', np.allclose(TransAll_list, TransAllThresh_list))
        return TransAll_list, TransAllThresh_list ,TransAllQuant_list
    
    def idct_decoder(self, TransAllQuant_list, Q, blocksize=8):
        h, w = TransAllQuant_list[0].shape
        B = blocksize
        DecAll = np.zeros((h, w, 3), np.uint8)
        for idx, channel in enumerate(TransAllQuant_list):
            channelrows = channel.shape[0]
            channelcols = channel.shape[1]
            blocksV = int(channelrows / B)
            blocksH = int(channelcols / B)
            back0 = np.zeros((channelrows, channelcols), np.uint8)
            for row in range(blocksV):
                for col in range(blocksH):
                    dequantblock = channel[row*B:(row+1)*B, col*B:(col+1)*B] * Q[idx]
                    currentblock = np.round(cv2.idct(dequantblock)) + 128 # inverse shiftign of the shift of the pixel values, sucht that their value range is [0,...,255].
                    currentblock[currentblock > 255] = 255
                    currentblock[currentblock < 0] = 0
                    back0[row*B:(row+1)*B, col*B:(col+1)*B] = currentblock
            back1 = cv2.resize(back0, (w, h)) # the subsampled chrominance channels are interpolated, using cv2.INTER_LINEAR in default. 
            DecAll[:, :, idx] = np.round(back1)
        return DecAll
    
    def split_frequency(self, Trans_list, blocksize=8):
        DctBlock_list = []
        B = blocksize
        for idx, channel in enumerate(Trans_list):
            channelrows = channel.shape[0]
            channelcols = channel.shape[1]
            blocksV = int(channelrows / B)
            blocksH = int(channelcols / B)
            DCT_block = np.zeros((blocksV, blocksH, B*B), np.float32)
            for row in range(blocksV):
                for col in range(blocksH):
                    currentblock = channel[row*B:(row+1)*B, col*B:(col+1)*B]
                    DCT_block[row, col] = currentblock.reshape(B*B)
            DctBlock_list.append(DCT_block)
        return DctBlock_list
    
    def upsample(self, DctBlock_list):
        h, w, c = DctBlock_list[0].shape
        DctUpAll = np.zeros((h, w, 3*c), np.float32)
        for idx, channel in enumerate(DctBlock_list):
            if idx == 0:
                DctUpAll[:, :, idx*c:(idx+1)*c] = channel
            else: 
                dct_block = cv2.resize(channel, (w,h))
                DctUpAll[:, :, idx*c:(idx+1)*c] = dct_block
        return DctUpAll

# class CityscapesEval():
#     # reference code: https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/cityscapes.py
#     def __init__(self, split='val', data_root=None, data_list=None):
#         assert split in ['val', 'test']
#         if not os.path.isfile(data_list):
#             raise (RuntimeError('Image list file do not exist:' + data_list + '\n'))
#         self.split = split

def _convert_to_label_id(result):
    """Convert trainId to id for cityscapes."""
    if isinstance(result, str):
        result = np.load(result)
    result_copy = result.copy()
    for trainId, label in CSLabels.trainId2label.items():
        result_copy[result == trainId] = label.id

    return result_copy

def results2img(results, data_list, data_root, save_dir, to_label_id):
    """Write the segmentation results to images.

    Args:
        results (list[list | tuple | ndarray]): Testing results of the
            dataset.
        imgfile_prefix (str): The filename prefix of the png files.
            If the prefix is "somepath/xxx",
            the png files will be named "somepath/xxx.png".
        to_label_id (bool): whether convert output to label_id for
            submission

    Returns:
        list[str: str]: result txt files which contains corresponding
        semantic segmentation images.
    """
    assert isinstance(results, list), 'results must be a list'
    list_read = open(data_list).readlines()
    assert len(list_read) == len(results)
    filenames = []
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        
        image_name = os.path.join(data_root, line_split[0])
        filenames.append(image_name)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    result_files = []
    for idx in range(len(results)):
        result = results[idx]
        if to_label_id:
            result = _convert_to_label_id(result)
        filename = filenames[idx]
        basename = os.path.splitext(os.path.basename(filename))[0]

        png_filename = os.path.join(save_dir, f'{basename}.png')

        output = Image.fromarray(result.astype(np.uint8)).convert('P')
        palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
        for label_id, label in CSLabels.id2label.items():
            palette[label_id] = label.color

        output.putpalette(palette) # if you want to see te color result
        output.save(png_filename)
        result_files.append(png_filename)

    return result_files
    
def format_results(self, results, imgfile_prefix=None, to_label_id=True):
    """Format the results into dir (standard format for Cityscapes
    evaluation).

    Args:
        results (list): Testing results of the dataset.
        imgfile_prefix (str | None): The prefix of images files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Default: None.
        to_label_id (bool): whether convert output to label_id for
            submission. Default: False

    Returns:
        tuple: (result_files, tmp_dir), result_files is a list containing
            the image paths, tmp_dir is the temporal directory created
            for saving json/png files when img_prefix is not specified.
    """

    assert isinstance(results, list), 'results must be a list'
    assert len(results) == len(self), (
        'The length of results is not equal to the dataset len: '
        f'{len(results)} != {len(self)}')

    if imgfile_prefix is None:
        tmp_dir = tempfile.TemporaryDirectory()
        imgfile_prefix = tmp_dir.name
    else:
        tmp_dir = None
    result_files = self.results2img(results, imgfile_prefix, to_label_id)

    return result_files, tmp_dir
    
def evaluate(self,
            results,
            metric='mIoU',
            logger=None,
            imgfile_prefix=None,
            efficient_test=False):
    """Evaluation in Cityscapes/default protocol.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
        logger (logging.Logger | None | str): Logger used for printing
            related information during evaluation. Default: None.
        imgfile_prefix (str | None): The prefix of output image file,
            for cityscapes evaluation only. It includes the file path and
            the prefix of filename, e.g., "a/b/prefix".
            If results are evaluated with cityscapes protocol, it would be
            the prefix of output png files. The output files would be
            png images under folder "a/b/prefix/xxx.png", where "xxx" is
            the image name of cityscapes. If not specified, a temp file
            will be created for evaluation.
            Default: None.

    Returns:
        dict[str, float]: Cityscapes/default metrics.
    """

    eval_results = dict()
    metrics = metric.copy() if isinstance(metric, list) else [metric]
    if 'cityscapes' in metrics:
        eval_results.update(
            self._evaluate_cityscapes(results, logger, imgfile_prefix))
        metrics.remove('cityscapes')
    # if len(metrics) > 0:
    #     eval_results.update(
    #         super(CityscapesDataset,
    #               self).evaluate(results, metrics, logger, efficient_test))

    return eval_results

def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
    """Evaluation in Cityscapes protocol.

    Args:
        results (list): Testing results of the dataset.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
        imgfile_prefix (str | None): The prefix of output image file

    Returns:
        dict[str: float]: Cityscapes evaluation results.
    """
    msg = 'Evaluating in Cityscapes style'
    print(msg)

    result_files, tmp_dir = self.format_results(results, imgfile_prefix)

    if tmp_dir is None:
        result_dir = imgfile_prefix
    else:
        result_dir = tmp_dir.name

    eval_results = dict()
    print(f'Evaluating results under {result_dir} ...', logger=logger)

    CSEval.args.evalInstLevelScore = True
    CSEval.args.predictionPath = os.path.abspath(result_dir)
    CSEval.args.evalPixelAccuracy = True
    CSEval.args.JSONOutput = False

    seg_map_list = []
    pred_list = []

    # # when evaluating with official cityscapesscripts,
    # # **_gtFine_labelIds.png is used
    # for seg_map in mmcv.scandir(
    #         self.ann_dir, 'gtFine_labelIds.png', recursive=True):
    #     seg_map_list.append(os.path.join(self.ann_dir, seg_map))
    #     pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

    # eval_results.update(
    #     CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return eval_results
