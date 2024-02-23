#%%
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


# --------- function from the hw1 start ---------------
def get_differential_filter():

    # To do

    def filter_x(diff_along_x, padded_im, row, column):

        three_by_three = padded_im[row:row+3, column:column+3]
        current_p = three_by_three[1, 1]
        next_p = three_by_three[1, 2]

        change = next_p - current_p

        return change

    def filter_y(diff_along_y, padded_im, row, column):

        three_by_three = padded_im[row:row+3, column:column+3]
        current_p = three_by_three[1, 1]
        next_p = three_by_three[2, 1]

        change = next_p - current_p
        
        return change

    return filter_x, filter_y



def filter_image(im, filter):

    # To do

    im_norm = (im-np.min(im))/(np.max(im)-np.min(im))

    length = im.shape[0]
    width = im.shape[1]

    pads = np.zeros((length+2, width+2))
    pads[1:length+1,1:width+1] = im_norm
    padded_im = pads

    im_filtered = []

    diff_along_x = im_norm.copy()
    diff_along_y = im_norm.copy()

    filter_x = filter[0]
    filter_y = filter[1]

    for row in range(im.shape[0]):
        for column in range(im.shape[1]):
             
            diff_along_x[row, column] = filter_x(diff_along_x, padded_im, row, column) 


    for row in range(im.shape[0]):
        for column in range(im.shape[1]):
             
            diff_along_y[row, column] = filter_y(diff_along_x, padded_im, row, column)



    im_filtered.append(diff_along_x)
    im_filtered.append(diff_along_y)

    return im_filtered



def get_gradient(im_dx, im_dy):

    # To do

    im_dx_copied = im_dx.copy()
    im_dy_copied = im_dy.copy()

    grad_mag = (im_dx_copied**2+im_dy_copied**2)**(1/2)
    grad_angle = np.arctan((im_dy_copied/(im_dx_copied+0.0000001)))*180/np.pi

    for i in range(grad_angle.shape[0]):
        for j in range(grad_angle.shape[1]):
            
            if grad_angle[i][j]<0:
                grad_angle[i][j] += 180

    return grad_mag, grad_angle
# --------- function from the hw1 end ---------------







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

    print()
    print('Received a image for warping!')
    print()

    part_target = img

    img_cds = []


    for x_value in range(part_target.shape[1]):
        for y_value in range(part_target.shape[0]):

            img_cds.append([x_value, y_value, 1])


    img_cds = np.array(img_cds)

    inversed_cds = np.linalg.inv(A)@np.transpose(img_cds)

    img2 = np.zeros((output_size[0], output_size[1]))
    
    inversed_cds = np.transpose(inversed_cds)

    for i, j in zip(inversed_cds, img_cds):

        x_value = i[0]
        y_value = i[1]

        if x_value>0 and y_value>0:
            if int(x_value)<img2.shape[1] and int(y_value)<img2.shape[0]:
                img2[int(y_value), int(x_value)] = part_target[j[1], j[0]]




    img2_rows = []
    img2_columns = []
    img2_pixel_values = []


    for row in range(img2.shape[0]):
        for column in range(img2.shape[1]):
            img2_rows.append(row)
            img2_columns.append(column)
            img2_pixel_values.append(img2[row, column])

    img2_rows = np.array(img2_rows)
    img2_columns = np.array(img2_columns)
    img2_pixel_values = np.array(img2_pixel_values)


    zero_indices = np.where(img2_pixel_values == 0)[0]
    cur_num_zero = len(zero_indices)
    new_num_zero = 0

    print("Let's start interpolation.")
    print()

    while cur_num_zero!=new_num_zero and cur_num_zero>1000:

        cur_num_zero = len(zero_indices)

        # Create a grid of points for interpolation
        xi, yi = np.mgrid[min(img2_rows)-1:max(img2_rows), min(img2_columns)-1:max(img2_columns)]

        # Interpolate values using griddata with 'nearest' method
        zi = interpolate.griddata((img2_rows, img2_columns), img2_pixel_values, (xi, yi), method='nearest')
        zi = zi.flatten()

        # Replace zero values with interpolated values
        for idx in zero_indices:
            img2_pixel_values[idx] = zi[idx]


        Z = np.array(img2_pixel_values).reshape(1068, 777)
        
        zero_indices = np.where(img2_pixel_values == 0)[0]

        new_num_zero = len(zero_indices)

        print('After interpolation, the number of empty pixel is:', new_num_zero)
    
    else:
        Z = np.array(img2_pixel_values).reshape(1068, 777)
        print('Finished interpolation because there was no improvement.')


    img_warped = Z
        


    return img_warped





def align_image(template, target, A):
    # To do

    output_filter_x = filter_image(template, get_differential_filter())[0]
    output_filter_y = filter_image(template, get_differential_filter())[1]

 
    D = []

    for i_y in range(output_filter_x.shape[0]):
        for i_x in range(output_filter_x.shape[1]):

            D.append(np.array([output_filter_x[i_y][i_x]*i_x,
                          output_filter_x[i_y][i_x]*i_y,
                          output_filter_x[i_y][i_x],
                          output_filter_y[i_y][i_x]*i_x,
                          output_filter_y[i_y][i_x]*i_y,
                          output_filter_y[i_y][i_x]]))
            
    D = np.array(D)
    H = np.matmul(D.transpose(), D)

    
    while True:
        
        print()
        print('Current affine matrix:', A)
        warped_img = warp_image(target, A, template.shape)

        print()
        print('Here is the output.')
        print()
        
        plt.imshow(warped_img, cmap='gray')
        plt.axis('off')
        plt.show()

        error = warped_img - template

        plt.imshow(abs(error), cmap='jet')
        plt.axis('off')
        plt.show()

        D = []

        for i_y in range(output_filter_x.shape[0]):
            for i_x in range(output_filter_x.shape[1]):
                

                D.append(np.array([output_filter_x[i_y][i_x]*i_x*error[i_y][i_x],
                              output_filter_x[i_y][i_x]*i_y*error[i_y][i_x], 
                              output_filter_x[i_y][i_x]*error[i_y][i_x],
                              output_filter_y[i_y][i_x]*i_x*error[i_y][i_x], 
                              output_filter_y[i_y][i_x]*i_y*error[i_y][i_x],
                              output_filter_y[i_y][i_x]*error[i_y][i_x]]))
            

        F = np.sum(D, axis=0)


        change_p = np.linalg.inv(H)@F.reshape(6, 1)

        change_A = np.array([[1+change_p[0][0],change_p[1][0], change_p[2][0]],
                             [change_p[3][0],1+change_p[4][0], change_p[5][0]],
                             [0, 0, 1]])

        new_A = A@np.linalg.inv(change_A)
        norm_error = np.linalg.norm(change_p)


        print('Magnitude of required change of A is:', norm_error)


        if norm_error<100:
            print("Magnitude of required change of A is small enough to stop the iteration!")
            print("Let's use current affine transformation as a refined one!")
            break
        
        else:
            print("Magnitude of required change of A is still high... Let's update affine transformation")
            print("and keep doing experiment with new affine matrix.")
            A = new_A


    A_refined = new_A
    error = -100

    return A_refined, error




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
    
    plt.title('All matched Keypoints') # code I made
    plt.axis('off') # code I made
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
    
    plt.title('Inliers and Boundary') # code I made
    plt.axis('off')
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

    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    print('Found initial Affine Transformation Matrix using the template and the first target: ')
    print(A)

    print()
    print("Let's find optimized affine matrix for target 2")
    print()

    A_refined, error = align_image(template, target_list[1], A)
 
    
    # visualize_align_image(template, target_list[i], A, A_refined, errors)

    # A_list = track_multi_frames(template, target_list)
    # visualize_track_multi_frames(template, target_list, A_list)


 # %%
