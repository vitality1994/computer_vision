#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#%%
'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

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


def build_histogram(grad_mag, grad_angle, cell_size):

    # To do

    one_length_cell_size = cell_size

    ori_histo = np.zeros((6, int(grad_angle.shape[0]/one_length_cell_size),\
                        int(grad_angle.shape[1]/one_length_cell_size)))

    for start_point_y in range(int(grad_angle.shape[0]/one_length_cell_size)):
        for start_point_x in range(int(grad_angle.shape[1]/one_length_cell_size)):
            
            one_cell_angle = grad_angle[start_point_y*one_length_cell_size:\
                                    start_point_y*one_length_cell_size+one_length_cell_size, \
                                    start_point_x*one_length_cell_size:\
                                    start_point_x*one_length_cell_size+one_length_cell_size]
            
            one_cell_mag = grad_mag[start_point_y*one_length_cell_size:\
                                    start_point_y*one_length_cell_size+one_length_cell_size, \
                                    start_point_x*one_length_cell_size:\
                                    start_point_x*one_length_cell_size+one_length_cell_size]

            bin_1 = 0
            bin_2 = 0
            bin_3 = 0
            bin_4 = 0
            bin_5 = 0
            bin_6 = 0

                    
            for mag, angle in zip(np.nditer(one_cell_mag), np.nditer(one_cell_angle)):
                    
                    if (0<=angle and angle<5) or (165<=angle and angle<180):
                            bin_1+=mag
                    elif 15<=angle and angle<45:
                            bin_2+=mag
                    elif 45<=angle and angle<75:
                            bin_3+=mag
                    elif 75<=angle and angle<105:
                            bin_4+=mag
                    elif 105<=angle and angle<135:
                            bin_5+=mag
                    elif 135<=angle and angle<165:
                            bin_6+=mag
                            
            bins = [bin_1, bin_2, bin_3, bin_4, bin_5, bin_6]
            
            for num_bin in range(6):   
                    
                    ori_histo[num_bin, start_point_y, start_point_x]=bins[num_bin]

    return ori_histo


def get_block_descriptor(ori_histo, block_size):

    # To do

    block_size = block_size

    ori_histo_normalized = np.zeros((6*block_size**2, ori_histo.shape[1]-block_size+1, ori_histo.shape[2]-block_size+1))

    for start_point_y in range(ori_histo.shape[1]-block_size+1):
        for start_point_x in range(ori_histo.shape[2]-block_size+1):
            
            one_block = ori_histo[:, start_point_y:start_point_y+block_size,\
                                start_point_x:start_point_x+block_size]

            denominator = np.sqrt((one_block.flatten()**2).sum()+0.0000001)

            concatenated_hog = one_block.flatten()/denominator

            for num_bin in range(len(concatenated_hog)):   
                
                    ori_histo_normalized[num_bin, start_point_y, start_point_x]=concatenated_hog[num_bin]
    
    
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    
    # To do

    output_filter_x = filter_image(im, get_differential_filter())[0]
    output_filter_y = filter_image(im, get_differential_filter())[1]

    grad_mag = get_gradient(output_filter_x, output_filter_y)[0]
    grad_angle = get_gradient(output_filter_x, output_filter_y)[1]

    ori_histo = build_histogram(grad_mag, grad_angle, 8)

    hog = get_block_descriptor(ori_histo, 2)

    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):

    # To do

    y_length_template = I_template.shape[0]

    template_norm_desc = extract_hog(I_template)
    target_norm_desc = extract_hog(I_target)
    
    y_norm_length_template = template_norm_desc.shape[1]
    x_norm_length_template = template_norm_desc.shape[2]

    bounding_boxes = []

    for start_point_y in range(target_norm_desc.shape[1]-y_norm_length_template):
        for start_point_x in range(target_norm_desc.shape[2]-x_norm_length_template):
                
            box = target_norm_desc[:, start_point_y:start_point_y+y_norm_length_template,\
                            start_point_x:start_point_x+x_norm_length_template]
            
            a = template_norm_desc.flatten()    
            b = box.flatten()

            
            try:

                sim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            
            except:
                sim = 0

            if sim>0.65 :
                
                # 8 is size of cell defined above
                bounding_boxes.append([start_point_x*8, start_point_x*8+I_template.shape[0],\
                                        start_point_y*8, start_point_y*8+I_template.shape[1],
                                        sim])
                

    bounding_boxes_1 = np.array(bounding_boxes)
    bounding_boxes_2 = np.array(bounding_boxes)

    result_bbs = []

    for bb1 in bounding_boxes_1[1:]:
        bb_group = []
        
        for bb2 in bounding_boxes_2:


            x_left = max(bb1[0], bb2[0])
            x_right = min(bb1[1], bb2[1])
            y_bottom = min(bb1[3], bb2[3])
            y_top = max(bb1[2], bb2[2])

            if x_left<x_right and y_top<y_bottom:
                
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            else:
                 intersection_area = 0
            
            if intersection_area<y_length_template**2 and intersection_area>0:

                bb1_area = (bb1[1]-bb1[0]) * (bb1[3] - bb1[2])
                bb2_area = (bb2[1]-bb2[0]) * (bb2[3] - bb2[2])
                

                if intersection_area < float(bb1_area + bb2_area - intersection_area):

                    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        
                    if iou>0.5:
                        bb_group.append(bb2)
                        
            bb_group.append(bb1)

        list_sims = []

        for bbs in bb_group:
            list_sims.append(bbs[4])


        bb_with_max = bb_group[np.argmax(list_sims)]
        result_bbs.append((bb_with_max[0], bb_with_max[2], bb_with_max[4]))


        
    
    bounding_boxes = np.array(result_bbs)
    

    return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape
    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (0, 255, 0), 1, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()



#%%
if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)    
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)
    
    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.

# %%
