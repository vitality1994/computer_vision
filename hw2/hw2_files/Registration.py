#%%
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

#%%
def find_match(img1, img2):

    # To do

    x1 = []
    x2 = []

    template = img1
    target = img2

    sift = cv2.SIFT_create()

    kp_temp, des_temp = sift.detectAndCompute(template, None)
    kp_coordinates_temp = [keypoint.pt for keypoint in kp_temp]

    kp_target, des_target = sift.detectAndCompute(target, None)
    kp_coordinates_target = [keypoint.pt for keypoint in kp_target]

    for one_des_temp, one_kp_cd in zip(des_temp, kp_coordinates_temp):

        combined_des = np.append([one_des_temp], des_target, axis=0)

        nbrs = NearestNeighbors(n_neighbors=3).fit(combined_des)

        distances, indices = nbrs.kneighbors(combined_des)

        first_nbr_idx = indices[0][1]
        second_nbr_inx = indices[0][2]

        first_nbr_dist = distances[0][1]
        second_nbr_dist = distances[0][2]     

        
        if first_nbr_dist/second_nbr_dist < 0.7:
            x1.append([one_kp_cd[0], one_kp_cd[1]])
            x2.append([kp_coordinates_target[first_nbr_idx-1][0],
                       kp_coordinates_target[first_nbr_idx-1][1]])
            
        
    x1 = np.array(x1)
    x2 = np.array(x2)

    return x1, x2




def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):


    template = cv2.imread('./JS_template.jpg', 0)
    target = cv2.imread('./JS_target1.jpg', 0)

    # To do

    nums_inliers = []
    coeffs_list = []

    for num_iter in range(ransac_iter):

        xs_temp = []
        xs_target = []
        ys_temp = []
        ys_target = []

        randints = np.random.randint(0, len(x1), 3)

        x1_sampled = x1[randints]
        x2_sampled = x2[randints]


        for one_x1, one_x2 in zip(x1_sampled, x2_sampled):

            x_temp = one_x1[0]
            y_temp = one_x1[1]

            x_target = one_x2[0]
            y_target = one_x2[1]

            # xs_temp.append([x_temp, 1])
            # xs_target.append([x_target])

            # ys_temp.append([y_temp, 1])
            # ys_target.append([y_target])


            xs_temp.append(x_temp)
            xs_target.append(x_target)

            ys_temp.append(y_temp)
            ys_target.append(y_target)


        xs_temp = np.array(xs_temp)
        xs_target = np.array(xs_target)
        ys_temp = np.array(ys_temp)
        ys_target = np.array(ys_target)


        A_x = np.vstack([xs_temp, np.ones(len(xs_temp))]).T
        m_x, c_x = np.linalg.lstsq(A_x, xs_target, rcond=None)[0]

        A_y = np.vstack([ys_temp, np.ones(len(ys_temp))]).T
        m_y, c_y = np.linalg.lstsq(A_y, ys_target, rcond=None)[0]

        
        inliers_temp = []
        inliers_target = [] 

        for one_cd_temp, one_cd_target in zip(x1, x2):
            
            d_x_from_line = abs(one_cd_target[0]-m_x*one_cd_temp[0]-c_x)
            d_y_from_line  = abs(one_cd_target[1]-m_y*one_cd_temp[1]-c_y)

            if d_x_from_line < ransac_thr and d_y_from_line < ransac_thr:

                inliers_temp.append(one_cd_temp)
                inliers_target.append(one_cd_target)
        

        num_inlier = len(inliers_temp)
        print(num_inlier)

        nums_inliers.append(num_inlier)
        coeffs_list.append([m_x, c_x, m_y, c_y])

        inliers_temp = np.array(inliers_temp)
        inliers_target = np.array(inliers_target)

        visualize_find_match(template, target, inliers_temp, inliers_target)


    idx_max_inliners = np.argmax(nums_inliers)
    m_x, c_x, m_y, c_y = coeffs_list[idx_max_inliners]


    inliers_temp = []
    inliers_target = [] 

    for one_cd_temp, one_cd_target in zip(x1, x2):
        
        d_x_from_line = abs(one_cd_target[0]-m_x*one_cd_temp[0]-c_x)
        d_y_from_line  = abs(one_cd_target[1]-m_y*one_cd_temp[1]-c_y)


        if d_x_from_line < ransac_thr and d_y_from_line < ransac_thr:

            inliers_temp.append(one_cd_temp)
            inliers_target.append(one_cd_target)
   

    print(len(inliers_temp))


    inliers_temp = np.array(inliers_temp)
    inliers_target = np.array(inliers_target)

    ### 3x3 affine transformation should be implemented



    return inliers_temp, inliers_target, A





def warp_image(img, A, output_size):
    # To do
    return img_warped


def align_image(template, target, A):
    # To do
    return A_refined


def track_multi_frames(template, img_list):
    # To do
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')

    x2_only_xs = list(map(lambda x: x[0], x2))
    x2_only_ys = list(map(lambda x: x[1], x2))

    x_max_x2 = max(x2_only_xs)
    x_min_x2 = min(x2_only_xs)
    y_max_x2 = max(x2_only_ys)
    y_min_x2 = min(x2_only_ys)

    plt.plot([x_min_x2, x_max_x2, x_max_x2, x_min_x2, x_min_x2], [y_min_x2, y_min_x2, y_max_x2, y_max_x2, y_min_x2], 'r-')

    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

#%%
if __name__ == '__main__':

    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target) 


    # ----------------- #3
    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)


    # ----------------- #4
    ransac_thr = 150
    ransac_iter = 5

    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_find_match(template, target_list[0], A[0], A[1])

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)



# %%
