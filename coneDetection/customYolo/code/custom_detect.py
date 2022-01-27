
"""
Custom detection script.
Modified version of Yolov5 detect.

"""

import argparse
import math
import os
import sys
from pathlib import Path
from scipy.linalg import block_diag
from scipy.optimize import curve_fit
from scipy.linalg import inv
import configparser

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from helpers import pretty_str, reshape_z
from kalman_filter import KalmanFilter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh, updateRoiCoordinates, predictRoiPosition)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# Load settings file
settingsFilePath = "./settings.ini"
config = configparser.ConfigParser()
config.read(settingsFilePath)

# TRUE -> print ROI and control points
# FALSE -> no prints
drawDetails = True


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()


    old_nearestXY_blu = [0,0]
    old_nearestXY_yellow = [0,0]
    nearestBlu_wh = [0,0]
    nearestYel_wh = [0,0]
    oldMiddlePoint = [0,0]
    newRoi_xxyy = []
    old_velocity_x = 0
    old_velocity_y = 0

    # Model speed
    velocity = config.getint('Kalman','velocity')

    # Load initial ROI values from settings file
    ROI_width = config.getfloat('ROI_parameters','ROI_width')
    ROI_height = config.getfloat('ROI_parameters','ROI_height')
    predictedROI = [config.getfloat('ROI_parameters','ROI_middlePoint_x'), config.getfloat('ROI_parameters','ROI_middlePoint_y')]


    middlePointsArray = np.empty([10, 2], dtype=int)
    middlePointsIndex = 0
    middlePointsCounter = 0
    avgMiddlePoint_x = 0
    avgMiddlePoint_y = 0

    plot_times = []
    plot_inference_times = []

    # Kalman Filter definition
    f = KalmanFilter(dim_x=2, dim_z=2)
    f.x = np.array([[682, 150]]).T
    f.F = np.array([ [1.,1.], [0.,1.],])
    f.H = np.array([[1, 0],[0, 1]])
    f.R = np.eye(2) * 0.35**2
    #f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    f.P = np.eye(2) * 500.


    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s, original_img, padx, pady in dataset:

        t1 = time_sync()
        #print((t1 - precFrameTime)*1000)
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2


        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        nearestXY_blu = [0,0]
        nearestXY_yellow = [0,0]
        middlePoint = []
        remainingCones = []


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(original_img, line_width=line_thickness, example=str(names))



            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], original_img.shape, padx, pady).round()
            
                # Remove detection if the cone is too far (depending on dimension respect to the nearest of the same color)
                # Tune the % of the width (ex: 30%)
                for y, elem in enumerate(det[:, :].tolist()):
                    if ((elem[5] == 0 and (elem[2] - elem[0] < 0.3*nearestBlu_wh[0])) or (elem[5] == 2 and (elem[2] - elem[0] < 0.3*nearestYel_wh[0])) or (elem[5] == 1 and ((elem[2] - elem[0] < 0.3*nearestYel_wh[0]) or (elem[2] - elem[0] < 0.3*nearestBlu_wh[0])))):
                        continue
                    else:
                        remainingCones.append(det[y, :].tolist())
                if(remainingCones and device.type != 'cpu'):
                    det = torch.tensor(remainingCones, device=torch.device('cuda:0'))
            
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # blu 0
                # yellow 2
                # orange 1
                count_blu = 0
                count_yellow = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if (c==0):
                            annotator.box_label(xyxy, label, color=[255, 0, 0])
                        if (c==2):
                            annotator.box_label(xyxy, label, color=[0, 255, 255])
                        if (c==1):
                            annotator.box_label(xyxy, label, color=[26, 140, 255])

                        # Update nearest cones (yellow and blu)
                        if (cls == 0 and xyxy[3].item() > nearestXY_blu[1]): 
                            nearestXY_blu[0]=xyxy[2].item() - (xyxy[2].item() - xyxy[0].item())/2
                            nearestXY_blu[1]=xyxy[3].item()
                            nearestBlu_wh = [xyxy[2].item() -  xyxy[0].item(), xyxy[3].item() -  xyxy[1].item()]
                            count_blu += 1

                        if (cls == 2 and xyxy[3].item() > nearestXY_yellow[1]): 
                            nearestXY_yellow[0]=xyxy[2].item() - (xyxy[2].item() - xyxy[0].item())/2
                            nearestXY_yellow[1]=xyxy[3].item()
                            nearestYel_wh = [xyxy[2].item() -  xyxy[0].item(), xyxy[3].item() -  xyxy[1].item()]
                            count_yellow += 1
                        

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    

                # Check if a color is not present and consider the middlePoint of the nearest cone of the precedent frame
                if (count_blu == 0):    
                    nearestXY_blu = old_nearestXY_blu
                if (count_yellow == 0): 
                    nearestXY_yellow = old_nearestXY_yellow

                # Don't consider the middlePoint of cones too far (blu and yellow) or too far from camera (check con distance with the 0 of the y)
                if ((nearestXY_blu[1] - nearestXY_yellow[1] > 40) and (nearestXY_yellow[1] < 0.2*original_img.shape[0])): 
                    nearestXY_yellow = old_nearestXY_yellow
                if ((nearestXY_yellow[1] - nearestXY_blu[1] > 40) and (nearestXY_blu[1] < 0.2*original_img.shape[0])): 
                    nearestXY_blu = old_nearestXY_blu



            if (nearestXY_blu[1] >= nearestXY_yellow[1]):
                middlePoint = nearestXY_blu[0] + (nearestXY_yellow[0] - nearestXY_blu[0])/2, nearestXY_yellow[1] + (nearestXY_blu[1] - nearestXY_yellow[1])/2
                middlePointsCounter += 1
            else:
                middlePoint = nearestXY_blu[0] + (nearestXY_yellow[0] - nearestXY_blu[0])/2, nearestXY_blu[1] + (nearestXY_yellow[1] - nearestXY_blu[1])/2
                middlePointsCounter += 1

            
            # Kalman Filter

            # Calculate observation with a median filter
            if (middlePointsIndex == 10): middlePointsIndex = 0

            if (middlePoint == [0,0]):
                middlePoint = oldMiddlePoint

            middlePointsArray[middlePointsIndex] = middlePoint
            middlePointsIndex += 1

            if (middlePointsCounter >= 10):
                middlePointsSum_x = middlePointsArray[:,0].sum()
                middlePointsSum_y = middlePointsArray[:,1].sum()
                nRowsCols = middlePointsArray.shape[0]
                avgMiddlePoint_x = middlePointsSum_x/nRowsCols
                avgMiddlePoint_y = middlePointsSum_y/nRowsCols
                z = np.array([[avgMiddlePoint_x], [avgMiddlePoint_y]])
            else:
                z = np.array([[middlePoint[0]], [middlePoint[1]]])



            '''# observation on center of ROI
            if (middlePoint == [0,0]):
                z = np.array([[oldMiddlePoint[0]], [oldMiddlePoint[1]]])
            else:
                z = np.array([[middlePoint[0]], [middlePoint[1]]])'''

            # Calculate movement of midlle points on the axes
            
            #if (middlePoint != [0,0]):
            if (middlePoint[1] > oldMiddlePoint[1]):
                dx = -(middlePoint[0] - oldMiddlePoint[0])
                dy = -(middlePoint[1] - oldMiddlePoint[1])
            else:
                dx = middlePoint[0] - oldMiddlePoint[0]
                dy = middlePoint[1] - oldMiddlePoint[1]

            time = (time_sync() - t1)*1000
            if (count_blu != 0 and count_yellow != 0):
                if (dx < 0): 
                    velocity_x = -velocity
                    old_velocity_x = velocity_x
                else:
                    velocity_x = velocity
                    old_velocity_x = velocity_x
                if (dy < 0): 
                    velocity_y = -velocity
                    old_velocity_y = velocity_y
                else:
                    velocity_y = velocity
                    old_velocity_x = velocity_x
            else:
                velocity_x = old_velocity_x
                velocity_y = old_velocity_y


            f.predict()
            predictedVal = f.x
            f.update(z)

            # Update velocity
            f.x[0] = f.x[0] + (velocity_x*time)
            f.x[1] = f.x[1] + (velocity_y*time)

            # move center of ROI on Kalman's result
            predictedROI = [int(f.x[0].item()), int(f.x[1].item())]

            #predictedROI = ext_xy
            newRoi_xxyy = updateRoiCoordinates(predictedROI, ROI_width, ROI_height, original_img.shape)
            

            if (newRoi_xxyy):
                dataset.updateROI(newRoi_xxyy[0], newRoi_xxyy[1])
                
                    
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            plot_times.append(t3)
            plot_inference_times.append(t3-t2)

            # Stream results
            im0 = annotator.result()

            if (drawDetails):
                # Draw semi-transparent ROI
                blk = np.zeros(im0.shape, np.uint8)
                if (newRoi_xxyy):
                    cv2.rectangle(blk, (int(newRoi_xxyy[0][0]), int(newRoi_xxyy[1][0])), (int(newRoi_xxyy[0][1]), int(newRoi_xxyy[1][1])), (0, 255, 0), cv2.FILLED)
                im0 = cv2.addWeighted(im0, 1.0, blk, 0.25, 1)

                # Draw circle on mid points of nearest blu and yellow cones
                im0 = cv2.circle(im0, (int(nearestXY_blu[0]), int(nearestXY_blu[1])), 3, (0, 255, 0), 2)
                im0 = cv2.circle(im0, (int(nearestXY_yellow[0]), int(nearestXY_yellow[1])), 3, (0, 255, 0), 2)

                # Draw circle on center point among nearest blu and yellow cone
                im0 = cv2.circle(im0, (int(middlePoint[0]), int(middlePoint[1])), 3, (0, 0, 255), 2)
                im0 = cv2.circle(im0, (int(avgMiddlePoint_x), int(avgMiddlePoint_y)), 3, (0, 230, 230), 2)

                # Kalman after update
                im0 = cv2.circle(im0, (int(f.x[0].item()), int(f.x[1].item())), 5, (255, 0, 0), 2)


            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            old_nearestXY_blu = nearestXY_blu
            old_nearestXY_yellow = nearestXY_yellow

            if (middlePoint != [0,0]):
                oldMiddlePoint = middlePoint
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.3fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    y_mean = [np.mean(plot_inference_times) for i in plot_inference_times]
    ax = plt.subplot()
    data_line = ax.plot(plot_times, plot_inference_times, label='Data')
    mean_line = ax.plot(plot_times, y_mean, label='Mean', linestyle='--')
    legend = ax.legend(loc='upper right')
    
    #plt.savefig('../volume/TesiMagistrale/inference_test/results/finalResults/RTX2070S_oldVideo.png')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

