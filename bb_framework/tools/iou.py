import numpy as np

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def calculate_iou_matrix(output1,output2):
    iou_matrix = np.zeros((len(output1["pred_boxes"]),len(output2["pred_boxes"])))
    # calculate IoUs between prediction and ground truths, and store it in iou_matrix
    for j,(box_pred1,class_pred1) in enumerate(zip(output1["pred_boxes"],output1["pred_classes"])):
        for i,(box_pred2,class_pred2) in enumerate(zip(output2["pred_boxes"],output2["pred_classes"])):
            if class_pred1 != class_pred2:
                iou_matrix[j,i] = None
                continue
            
            box_pred_dic1 = {'x1':box_pred1[0],'y1':box_pred1[1]
                            ,'x2':box_pred1[0]+box_pred1[2],'y2':box_pred1[1]+box_pred1[3]}
            box_pred_dic2 = {'x1':box_pred2[0],'y1':box_pred2[1]
                            ,'x2':box_pred2[0]+box_pred2[2],'y2':box_pred2[1]+box_pred2[3]}
            
            iou = get_iou(box_pred_dic1,box_pred_dic2)
            iou_matrix[j,i] = iou
    return iou_matrix

def get_iou_max_matrix(iou_matrix):
    # calculate the max IoUs per row and per column
    iou_col_max_index = np.argmax(iou_matrix,0)
    iou_col_max = np.amax(iou_matrix,0)
    iou_row_max = np.amax(iou_matrix,1)
    #copy iou_matrix
    iou_matrix_copy = iou_matrix.copy()
    
    iou_thresh = 0.1
    # determine max in iou_matrix row and column
    for col,row in enumerate(iou_col_max_index):
        if iou_matrix[row][col] > iou_thresh and \
        iou_matrix[row][col] == iou_col_max[col] and \
        iou_matrix[row][col] == iou_row_max[row]:
            iou_matrix_copy[row][col] = None
    return iou_matrix_copy
