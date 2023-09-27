import cv2

def uvuv2uvwh(uvuv):
    center_u = (uvuv[0] + uvuv[2])/2
    center_v = (uvuv[1] + uvuv[3])/3
    w = uvuv[2] - uvuv[0]
    h = uvuv[3] - uvuv[1]
    return (center_u, center_v, w, h)

def plot_one_box(uvuv, img, color = (0, 200, 0), target = False):
    uv1 = (int(uvuv[0]), int(uvuv[1]))
    uv2 = (int(uvuv[2]), int(uvuv[3]))
    if target:
        color = (0, 0, 255)
    cv2.rectangle(img, uv1, uv2, color, 1, cv2.LINE_AA)

def update_trace_list(box_center, trace_list, max_list_len = 50):
    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
    else:
        trace_list.pop(0)
        trace_list.append(box_center)
    return trace_list

def draw_trace(img, trace_list):
    for i, item in enumerate(trace_list):

        if i < 1:
            continue
        cv2.line(
            img,
            (trace_list[i][0], trace_list[i][1]),
            (trace_list[i-1][0], trace_list[i-1][1]),
            (255, 255, 0),
            3
        )

def cal_iou(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # calculate the cover
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
    # calculate the intersection
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1., 0)
    inter_w = max(xmax - xmin + 1., 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection
    # calculate iou
    iou = intersection/ union
    return iou

def cal_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) // 2, (box1[1], box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1], box2[3]) // 2)
    dis = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) **0.5
    return dis

def uvwh2uvuv(uvwh):
    u1 = uvwh[0] - uvwh[2]//2
    v1 = uvwh[1] - uvwh[3]//2
    u2 = uvwh[0] + uvwh[2]//2
    v2 = uvwh[1] + uvwh[2]//2
    return [u1, v1, u2, v2]

if __name__ == "main":
    box1 = [15, 15, 25, 25]
    box2 = [17, 8, 27, 18] 
    iou = cal_iou(box1, box2)
    box1.pop(0)
    box1.append(555)
    print(box1)