
import numpy as np
import cv2
import math



def showImage(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)


def showVideo(video):
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0,video_length - 1):
        ret, frame = video.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def computeResiduals(frame, bg_model):
    residual_image = np.zeros((frame.shape[0], frame.shape[1]), float)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            res = float(frame[i,j,0]) - bg_model[i,j]
            residual_image[i,j] = res
    return residual_image


def generateMeanBackground(video, from_i, to_i):
    frame_height = video[0].shape[1]
    frame_width = video[0].shape[0]
    numFrames = abs(from_i - to_i)
    mean_background = np.zeros((frame_width, frame_height), float)
    for i in range(from_i, to_i):
        frame = video[i]
        for j in range(0, mean_background.shape[0]):
            for k in range(0, mean_background.shape[1]):
                mean_background[j,k] += frame[j,k,0] 
    for i in range(0, mean_background.shape[0]):
        for j in range(0, mean_background.shape[1]):
            mean_background[i,j] = mean_background[i,j]/numFrames
    return (mean_background)


def generateMedianBackground(video, from_i, to_i):
    frame_height = video[0].shape[1]
    frame_width = video[0].shape[0]
    numFrames = abs(from_i - to_i)
    median_background = np.zeros((frame_width, frame_height), float)
    for i in range(0, median_background.shape[0]):
        for j in range(0, median_background.shape[1]):
            values_over_time = list()
            for k in range(from_i, to_i):
                frame = video[k]
                values_over_time.append(frame[i,j,0])
            values_over_time.sort()
            median_background[i,j] = values_over_time[math.ceil(len(values_over_time)/2)]

    return (median_background)



def umbralize(img, thresh):
    mask = np.zeros((img.shape[0], img.shape[1]), int)
    arr = img.ravel()
    min_val = np.min(arr)
    max_val = np.max(arr)
    den = max_val - min_val
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            residual = pow(img[i,j],2)
            if den != 0:
                residual = ((residual - min_val)/(max_val - min_val)) * (1 - 0) + 0
                if residual > thresh:
                    mask[i,j] = 255
                else:
                    mask[i,j] = 0
            else:
                mask[i,j] = img[i,j]
    return mask


def movement_detection(video, ground_truth, num_frames, background_substraction, spacing):
    print('PREPARING MOVEMENT DETECTION')
    thresh = 0.98
    print('THRESH = ', thresh)
    print('FRAME SPACE BETWEEN EACH GENERATED MODEL  = ', spacing)
    print('\n-------------------------------------------------------------')
    print('\n\n-----------INITIALIAZING MOVEMENT DETECTION...-------.------')
    print('\n\n-----------------------------------------------------------\n')
    video_length = len(video)
    background_model = background_substraction(video, 0, num_frames)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    accumulator = 0
    print_flag = True
    for frame_index in range(num_frames, video_length - 1):
        accumulator += 1
        frame = video[frame_index]
        residual_frame = computeResiduals(frame, background_model)
        movement_mask = umbralize(residual_frame, thresh)
        if accumulator > spacing: 
            print('Generated New Background Model...\n')
            background_model = background_substraction(video, frame_index - num_frames + 1, frame_index)
            accumulator = 0
        frame_GT = umbralize_GT(ground_truth[frame_index])
        true_positives, true_negatives, false_positives, false_negatives  = compute_ROC_metrics(movement_mask, frame_GT, true_positives, true_negatives, false_positives, false_negatives)
        if accumulator > 350 and print_flag == True: 
            print('Se han escrito resultados provisionales en la carpeta del proyecto...\n\n')
            cv2.imwrite('frame_vid.png ' ,frame)
            cv2.imwrite('bg.png ' ,background_model)
            cv2.imwrite('mask.png ' ,movement_mask)
            cv2.imwrite('res.png ' ,residual_frame)
            cv2.imwrite('frame_GT.png ' ,frame_GT)                  
            input('Press enter to continue....\n')
            print_flag = False
       
    print('\nDone.\n')
    print('\n-------------------------------------------------------------')
    print('\n-----------------GENERATING ROC METRICS----------------------')
    print('\n-------------------------------------------------------------')
    PDR = 0
    NDR = 0
    P = 0
    F = 0
    print(true_negatives, true_positives, false_negatives, false_positives)
    if true_positives + false_negatives != 0:
        PDR = true_positives / (true_positives + false_negatives)
    if false_negatives + false_positives != 0:
        NDR = true_negatives/ (true_negatives + false_positives)
    if true_positives + false_positives != 0:
        P = true_positives/(true_positives + false_positives)
    if PDR + P != 0:
        F = ( 2 * PDR * P)/ (PDR + P)
    print('PDR = ', PDR)
    print('NDR = ', NDR)
    print('P = ', P)
    print('F = ', F)
    print('\nDone...\n')



def compute_ROC_metrics(movement_mask, frame_GT , TP, TN, FP, FN):
    for i in range(0 , movement_mask.shape[0]):
        for j in range(0, movement_mask.shape[1]):
            if movement_mask[i,j] == 255 and frame_GT[i,j] == 255:
                TP += 1
            elif  movement_mask[i,j] == 0 and frame_GT[i,j] == 0:
                TN += 1
            elif movement_mask[i,j] == 255 and frame_GT[i,j] == 0:
                FP += 1
            elif movement_mask[i,j] == 0 and frame_GT[i,j] == 255:
                FN += 1
    return TP,TN,FP,FN


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def umbralize_GT(GT):
    new_frame = np.zeros((GT.shape[0], GT.shape[1]), int)
    for i in range(0,GT.shape[0]):
        for j in range(0, GT.shape[1]):
            if GT[i,j,0] > 0:
                new_frame[i,j] = 255
            else:
                new_frame[i,j] = 0
    return new_frame


def showFrame(video, index):
    video.set(cv2.CAP_PROP_POS_FRAMES,index)
    showImage(video.read()[1])

def loadVideo(path):
    cap = list()
    video = cv2.VideoCapture(path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0,video_length - 1):
        ret, frame = video.read()
        cap.append(frame)
    video.release()
    return cap

def main():
    video = loadVideo('togray.avi')
    ground_truth = loadVideo('highway_GT_460to869.avi')
    num_frames = 20
    mean_BG_subs = generateMeanBackground
    median_BG_subs = generateMedianBackground
    spacing = int(input('Cuantos frames recorrer antes de la generación de cada modelo de fondo?\n'))
    if spacing == 0: spacing = 1000 
    if input('\n¿Que modelo de fondo utilizar? (contestar usando los indices sin punto)\n\n1.Media\n2.Mediana\n\n') == '1':
        print('\n\nCOMPUTING MOVEMENT DETECTION WITH MEAN BACKGROUND: \n')
        movement_detection(video, ground_truth, num_frames, mean_BG_subs, spacing)
    else:
        print('\n\nCOMPUTING MOVEMENT DETECTION WITH MEDIAN BACKGROUND: \n')
        movement_detection(video, ground_truth, num_frames, median_BG_subs,spacing)
    


main()