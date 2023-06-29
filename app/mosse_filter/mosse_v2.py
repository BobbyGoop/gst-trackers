import numpy as np
import cv2
import os
from gst_utils.cv_utils import linear_mapping, pre_process, random_warp
import sys


class MosseFilterV2:
    def __init__(self, args, video_path):
        # get arguments..
        self.args = args
        self.video_path = video_path
        # get the img lists...
        # self.frame_lists = self._get_img_lists(self.video_path)
        # self.frame_lists.sort()

    def start_tracking(self):
        # Read video
        video = cv2.VideoCapture(self.video_path)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # Draw the box
        init_gt = cv2.selectROI('demo', frame, False, False)
        init_gt = np.array(init_gt).astype(np.int64)

        # get the frame of the first frame... (read as gray scale frame...)
        # init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)

        # Train filter on first frame
        # Start to draw the gaussian response...
        response_map = self._get_gauss_response(init_frame, init_gt)

        # Start to create the training set and get the goal..
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        G = np.fft.fft2(g)

        # start to do the pre-training...
        Ai, Bi = self._pre_training(fi, G)

        Ai = self.args.lr * Ai
        Bi = self.args.lr * Bi
        pos = init_gt.copy()
        clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)

        # Start tracking (from frame 1 to N)
        # for idx in range(len(self.frame_lists)):
        while True:
            # Read a new frame
            ok, current_frame = video.read()
            if not ok:
                break

            # Start timer
            timer = cv2.getTickCount()

            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            Hi = Ai / Bi
            fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
            Gi = Hi * np.fft.fft2(fi)
            gi = linear_mapping(np.fft.ifft2(Gi))
            # find the max pos...
            max_value = np.max(gi)
            max_pos = np.where(gi == max_value)
            dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
            dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

            # update the position...
            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy

            # trying to get the clipped position [xmin, ymin, xmax, ymax]
            clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
            clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
            clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
            clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
            clip_pos = clip_pos.astype(np.int64)

            # get the current fi..
            fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
            # online update...
            Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
            Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw box
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)

            # Display tracker type on frame
            cv2.putText(frame, "MOSSE Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", current_frame)

            # visualize the tracking process...

            # cv2.imshow('demo', current_frame)
            # cv2.waitKey(100)
            # # if record... save the frames..
            # if self.args.record:
            #     frame_path = 'record_frames/' + self.video_path.split('/')[1] + '/'
            #     if not os.path.exists(frame_path):
            #         os.mkdir(frame_path)
            #     cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    def _pre_training(self, init_frame, G):
        """ Pre-train the filter on the first frame... """
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi

    def _get_gauss_response(self, img, gt):
        """Get the ground-truth gaussian response..."""
        # get the shape of the frame..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    @staticmethod
    def _get_img_lists(img_path):
        """Extract the frame list"""
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list

    @staticmethod
    def _get_init_ground_truth(img_path):
        """Get the first ground truth of the video"""
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]
