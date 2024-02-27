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

    # To do

    x1_for_affine = []

    for i in x1:
        x1_for_affine.append([i[0], i[1], 1])
    
    x1_for_affine = np.array(x1_for_affine)

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

        nums_inliers.append(num_inlier)
        coeffs_list.append([m_x, c_x, m_y, c_y])

        inliers_temp = np.array(inliers_temp)
        inliers_target = np.array(inliers_target)


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
   

    inliers_temp = np.array(inliers_temp)
    inliers_target = np.array(inliers_target)
    

    inliers_temp = np.array(list(map(lambda x: [x[0], x[1], 1], inliers_temp)))
    inliers_target = np.array(list(map(lambda x: [x[0], x[1], 1], inliers_target)))



    A = np.linalg.inv(np.transpose(inliers_temp)@inliers_temp)@np.transpose(inliers_temp)@inliers_target

    A = np.transpose(A)

    return A




def warp_image(img, A, output_size):

    # To do

    # Vectorize the creation of coordinates
    y_indices, x_indices = np.indices((img.shape[0], img.shape[1]))
    img_cds = np.stack([x_indices, y_indices, np.ones_like(x_indices)], axis=-1).reshape(-1, 3).T

    # Apply the transformation
    inversed_cds = np.linalg.inv(A) @ img_cds

    # Round the coordinates and clip to valid indices
    inversed_cds = np.round(inversed_cds).astype(int)
    valid_indices = (inversed_cds[0] >= 0) & (inversed_cds[0] < output_size[1]) & \
                    (inversed_cds[1] >= 0) & (inversed_cds[1] < output_size[0])

    # Create an empty image with the desired output size
    img_warped = np.zeros(output_size)

    # Use advanced indexing to map the valid coordinates
    img_warped[inversed_cds[1][valid_indices], inversed_cds[0][valid_indices]] = img[img_cds[1][valid_indices], img_cds[0][valid_indices]]

    # Check if interpolation is needed
    if np.any(img_warped == 0):

        # Interpolate only the zero values
        zero_indices = np.stack(np.nonzero(img_warped == 0), axis=-1)
        non_zero_indices = np.stack(np.nonzero(img_warped), axis=-1)
        img_warped[img_warped == 0] = interpolate.griddata(
            non_zero_indices, img_warped[img_warped != 0],
            zero_indices, method='nearest'
        )
        
    return img_warped



def align_image(template, target, A):

    # To do


    # -----------------------------------------------------------------------------
    # --------------- Implementation of Sobel filter ------------------------------

    Gx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    Gy = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

    [rows, columns] = np.shape(template)  # we need to know the shape of the input grayscale image

    padded_template = np.pad(template, (1, 1), 'edge')

    # Now we "sweep" the image in both x and y directions and compute the output
    gx = np.zeros(shape=(rows, columns))
    gy = np.zeros(shape=(rows, columns))

    for i in range(0, padded_template.shape[0]-2):
        for j in range(0, padded_template.shape[1]-2):
            gx[i][j] = np.sum(np.multiply(Gx, padded_template[i:i + 3, j:j + 3]))  # x direction
            gy[i][j] = np.sum(np.multiply(Gy, padded_template[i:i + 3, j:j + 3]))  # y direction


    # value multiplied to derivatives change 
    # the performance and speed of the codes.
    gx = gx*1/100
    gy = gy*1/100
    
    # Some data manipulation for better performance
    for i in range(gx.shape[0]):
        gx[i][0] = 0
        gx[i][-1] = 0

    for i in range(gy.shape[1]):
        gy[0][i] = 0
        gy[-1][i] = 0
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------   


    # -----------------------------------------------------------------------------
    # --------- Implementation of Inverse Compositional Images Alignment ----------
        
    # Compute indices (i_x, i_y) as grids
    Ix, Iy = np.meshgrid(np.arange(template.shape[1]), np.arange(template.shape[0]))

    # Vectorize computation of matrix D
    D = np.stack([gx * Ix, gx * Iy, gx, gy * Ix, gy * Iy, gy], axis=-1)
    
    # Initialize H as a zero matrix
    H = np.zeros((6, 6))
    for d in D.reshape(-1, 6):
        H += np.outer(d, d)
    
    num_itr = 0 
    errors_list = []

    while True:
        # Warp the image with the current estimate of A
        warped_img = warp_image(target, A, template.shape)
        
        if num_itr%5==0:
            
            plt.imshow(warped_img, cmap='gray')
            plt.show()

        # Compute the error image
        error = warped_img - template

        # Vectorize computation of matrix F
        F = np.einsum('ij,ijk->k', error, D)
        
        # Compute change in parameters
        change_p, _, _, _ = np.linalg.lstsq(H, F, rcond=None)
        
        # Update the transformation matrix A
        change_A = np.array([[1+change_p[0], change_p[1], change_p[2]],
                             [change_p[3], 1+change_p[4], change_p[5]],
                             [0, 0, 1]])
        
        # Compute the norm of the error
        norm_error = np.linalg.norm(error)
    
        num_itr += 1

        print('Magnitude of required change of A is:', norm_error)
        print('Number of iterations:', num_itr)

        errors_list.append(norm_error)

        # Check for convergence
        if num_itr > 30 or norm_error < 10000:
            break
    
        else:
            A = A @ np.linalg.inv(change_A)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------   
    
    A_refined = A
    errors = np.array(errors_list)

    return A_refined, errors
    



def track_multi_frames(template, img_list):

    # To do         
   
    x1, x2 = find_match(template, target_list[0])

    ransac_thr = 150
    ransac_iter = 5

    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    A_list = []

    for i in img_list:

        A_refined, error = align_image(template, i, A)
        visualize_align_image(template, i, A, A_refined, error)

        A_list.append(A_refined)


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
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    
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

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 150
    ransac_iter = 5

    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr, img_h=500)

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


# %%
