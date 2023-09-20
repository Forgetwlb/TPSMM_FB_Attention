import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
import matplotlib.pyplot as plt
import face_alignment
from evaluation.OpenFacePytorch.loadOpenFace import prepareOpenFace
from torch.autograd import Variable
from skimage.transform import resize

def reconstruction(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, kp_detector=kp_detector,
                        bg_predictor=bg_predictor, dense_motion_network=dense_motion_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    # net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()

    loss_list = []
    loss_mean = []
    loss_AKD = []
    loss_AED = []
    loss_deformed = []
    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            image_loss = []
            image_folder_loss = []
            kp_loss = []
            visualizations = []

            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                bg_params = None
                if bg_predictor:
                    bg_params = bg_predictor(source, driving)

                dense_motion = dense_motion_network(source_image=source, driving_image =driving, kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = bg_params,
                                                    dropout_flag = False)
                out = inpainting_network(source, driving, dense_motion)
                # del out['deformed_source']
                predict_image = out['prediction']
                kp_predict = kp_detector(predict_image)

                # # AKD 计算
                # gen_image = np.array(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]) * 255
                # kp_gen = fa.get_landmarks(gen_image)
                # kp_gen = kp_gen[0]
                # gt_image = np.array(np.transpose(driving.data.cpu().numpy(), [0, 2, 3, 1])[0]) * 255
                # kp_gt = fa.get_landmarks(gt_image)
                # kp_gt = kp_gt[0]
                # if kp_gen is not None:
                #     loss_AKD.append(np.mean(np.abs(kp_gt - kp_gen)))
                #
                # # AED 计算
                # cropped_img_gen = resize(gen_image, (96, 96))
                # id_gen_img = np.transpose(cropped_img_gen, (2, 0, 1))
                # with torch.no_grad():
                #     frame = Variable(torch.Tensor(id_gen_img)).cuda()
                #     frame = frame.unsqueeze(0)
                #     id_vec_gen = net(frame)[0].data.cpu().numpy()
                #
                # cropped_img_gt = resize(gt_image, (96, 96))
                # id_gt_img = np.transpose(cropped_img_gt, (2, 0, 1))
                # with torch.no_grad():
                #     frame2 = Variable(torch.Tensor(id_gt_img)).cuda()
                #     frame2 = frame2.unsqueeze(0)
                #     id_vec_gt = net(frame2)[0].data.cpu().numpy()
                #
                # loss_AED.append(np.sum(np.abs(id_vec_gt - id_vec_gen).astype(float) ** 2))

                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                out['kp_predict'] = kp_predict

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                # visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                #                                                                    driving=driving, out=out)
                # visualizations.append(visualization)
                image_save_dir = x['name'][0]

                pngdir = os.path.join(png_dir, image_save_dir)
                if not os.path.exists(pngdir):
                    os.makedirs(pngdir)


                loss_deformed.append(torch.abs(out['deformed'] - driving).mean().cpu().numpy())
                # loss = torch.abs(out['prediction'] - driving).mean().cpu().numpy()
                # loss_list.append(loss)

                # image_loss_array = torch.abs(out['prediction'] - driving).mean().cpu().numpy()
                # image_loss.append(np.mean(image_loss_array))
                # image_folder_loss.append(np.mean(image_loss_array))
                # imageio.imsave(os.path.join(pngdir, str(frame_idx + 1) + '.png'), visualization)

                # kp_loss_array = torch.abs((out['kp_driving']['fg_kp'] - out['kp_predict']['fg_kp']) * x['video'].shape[
                #     -1]).mean().cpu().numpy()
                # kp_loss.append(np.mean(kp_loss_array))

            # loss_list.append(image_folder_loss)
            # loss_tmp = np.mean(image_folder_loss)
            # loss_mean.append(np.mean(loss_tmp))
            # KP_list.append(sum(kp_loss))
            # # 绘制损失值曲线
            # plt.plot(image_loss)
            # # 添加标题和轴标签
            # plt.title(x['name'][0])
            # plt.xlabel('image_idx')
            # plt.ylabel('L1_loss')
            # # 保存图像
            # plt.savefig(pngdir + '/Table' + x['name'][0] + '.png')
            # plt.close()
            #
            # # 绘制关键点的差值
            # plt.plot(kp_loss)
            # # 添加标题和轴标签
            # plt.title(x['name'][0])
            # plt.xlabel('image_idx')
            # plt.ylabel('KP_L1')
            # # 保存图像
            # plt.savefig(pngdir + '/KP_Table' + x['name'][0] + '.png')
            # plt.close()
            # # print(np.mean(loss_list))
            # predictions = np.concatenate(predictions, axis=1)
            # imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

    # 绘制总损失曲线
    # plt.plot(loss_mean)
    # # 添加标题和轴标签
    # plt.title("Reconstruction all loss: %s" % np.mean(loss_mean))
    # plt.xlabel('image_folder_idx')
    # plt.ylabel('L1_loss')
    # # 保存图像
    # plt.savefig(pngdir + 'Reconstruction all loss.png')
    # plt.close()
    # print("Reconstruction loss: %s" % np.mean(loss_list))
    # print("AKD_loss: %s" % np.mean(loss_AKD))
    # print("AED_loss: %s" % np.mean(loss_AED))
    print("deformation: %s" % np.mean(loss_deformed))

