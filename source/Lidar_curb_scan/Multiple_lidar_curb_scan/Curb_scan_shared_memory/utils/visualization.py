# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=91)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
colors_custom = [(0,255,127), (240,248,255), (255,165,0), (255,192,203), (255,0,0),
                  (44, 160, 44), (113,102,255)]
colors_custom = [
                  ["AliceBlue",(240,248,255)],
                  ["YellowGreen",(154,205,50)], #Car
                  ["DeepPink",(255,20,147)], #Van
                  ["Gold",(255,215,0)],  #Truck
                  ["Orange",(255,165,0)],#Cyclist
                  ["OrangeRed",(255,69,0)],#Pedestrian
                  ["Orchid",(218,112,214)],#Person_sitting
                  ["Salmon",(250,128,114)], #Tram
                  ["Purple",(128,0,128)],#Misc
                  ["Yellow",(255,255,0)],#Sign
                  ["Indigo",(75,0,130)],#Traffic_cone
                  ["Tomato",(255,0,0)], #Traffic_light
                  ["Olive",(128,128,0)], #School_bus
                  ["Plum",(221,160,221)], #Bus
                  ["SandyBrown",(244,164,96)],]
named_colors_map = [
    ["AliceBlue",(240,248,255)],
    ["AntiqueWhite",(250,235,215)],
    ["Aqua",(0,255,255)],
    ["Aquamarine",(127,255,212)],
    ["Azure",(240,255,255)],
    ["Beige",(245,245,220)],
    ["Bisque",(255,228,196)],
    ["Black",(0,0,0)],
    ["BlanchedAlmond",(255,235,205)],
    # ["Blue",(0,0,255)],
    ["BlueViolet",(138,43,226)],
    # ["Brown",(165,42,42)],
    ["BurlyWood",(222,184,135)],
    ["CadetBlue",(95,158,160)],
    ["Chartreuse",(127,255,0)],
    ["Chocolate",(210,105,30)],
    ["Coral",(255,127,80)],
    ["CornflowerBlue",(100,149,237)],
    ["Cornsilk",(255,248,220)],
    ["Crimson",(220,20,60)],
    ["Cyan",(0,255,255)],
    # ["DarkBlue",(0,0,139)],
    # ["DarkCyan",(0,139,139)],
    # ["DarkGoldenRod",(184,134,11)],
    # ["DarkGray",(169,169,169)],
    # ["DarkGrey",(169,169,169)],
    # ["DarkGreen",(0,100,0)],
    # ["DarkKhaki",(189,183,107)],
    # ["DarkMagenta",(139,0,139)],
    # ["DarkOliveGreen",(85,107,47)],
    # ["DarkOrange",(255,140,0)],
    # ["DarkOrchid",(153,50,204)],
    # ["DarkRed",(139,0,0)],
    # ["DarkSalmon",(233,150,122)],
    # ["DarkSeaGreen",(143,188,143)],
    # ["DarkSlateBlue",(72,61,139)],
    # ["DarkSlateGray",(47,79,79)],
    # ["DarkSlateGrey",(47,79,79)],
    # ["DarkTurquoise",(0,206,209)],
    # ["DarkViolet",(148,0,211)],
    ["DeepPink",(255,20,147)],
    ["DeepSkyBlue",(0,191,255)],
    # ["DimGray",(105,105,105)],
    # ["DimGrey",(105,105,105)],
    ["DodgerBlue",(30,144,255)],
    ["FireBrick",(178,34,34)],
    ["FloralWhite",(255,250,240)],
    ["ForestGreen",(34,139,34)],
    ["Fuchsia",(255,0,255)],
    ["Gainsboro",(220,220,220)],
    ["GhostWhite",(248,248,255)],
    ["Gold",(255,215,0)],
    ["GoldenRod",(218,165,32)],
    # ["Gray",(128,128,128)],
    # ["Grey",(128,128,128)],
    ["Green",(0,128,0)],
    ["GreenYellow",(173,255,47)],
    ["HoneyDew",(240,255,240)],
    ["HotPink",(255,105,180)],
    ["IndianRed",(205,92,92)],
    ["Indigo",(75,0,130)],
    ["Ivory",(255,255,240)],
    ["Khaki",(240,230,140)],
    ["Lavender",(230,230,250)],
    ["LavenderBlush",(255,240,245)],
    ["LawnGreen",(124,252,0)],
    ["LemonChiffon",(255,250,205)],
    # ["LightBlue",(173,216,230)],
    # ["LightCoral",(240,128,128)],
    # ["LightCyan",(224,255,255)],
    # ["LightGoldenRodYellow",(250,250,210)],
    # ["LightGray",(211,211,211)],
    # ["LightGrey",(211,211,211)],
    # ["LightGreen",(144,238,144)],
    # ["LightPink",(255,182,193)],
    # ["LightSalmon",(255,160,122)],
    # ["LightSeaGreen",(32,178,170)],
    # ["LightSkyBlue",(135,206,250)],
    # ["LightSlateGray",(119,136,153)],
    # ["LightSlateGrey",(119,136,153)],
    # ["LightSteelBlue",(176,196,222)],
    # ["LightYellow",(255,255,224)],
    ["Lime",(0,255,0)],
    ["LimeGreen",(50,205,50)],
    ["Linen",(250,240,230)],
    ["Magenta",(255,0,255)],
    ["Maroon",(128,0,0)],
    ["MediumAquaMarine",(102,205,170)],
    ["MediumBlue",(0,0,205)],
    ["MediumOrchid",(186,85,211)],
    ["MediumPurple",(147,112,219)],
    ["MediumSeaGreen",(60,179,113)],
    ["MediumSlateBlue",(123,104,238)],
    ["MediumSpringGreen",(0,250,154)],
    ["MediumTurquoise",(72,209,204)],
    ["MediumVioletRed",(199,21,133)],
    ["MidnightBlue",(25,25,112)],
    ["MintCream",(245,255,250)],
    ["MistyRose",(255,228,225)],
    ["Moccasin",(255,228,181)],
    ["NavajoWhite",(255,222,173)],
    ["Navy",(0,0,128)],
    ["OldLace",(253,245,230)],
    ["Olive",(128,128,0)],
    ["OliveDrab",(107,142,35)],
    ["Orange",(255,165,0)],
    ["OrangeRed",(255,69,0)],
    ["Orchid",(218,112,214)],
    ["PaleGoldenRod",(238,232,170)],
    ["PaleGreen",(152,251,152)],
    ["PaleTurquoise",(175,238,238)],
    ["PaleVioletRed",(219,112,147)],
    ["PapayaWhip",(255,239,213)],
    ["PeachPuff",(255,218,185)],
    ["Peru",(205,133,63)],
    ["Pink",(255,192,203)],
    ["Plum",(221,160,221)],
    ["PowderBlue",(176,224,230)],
    ["Purple",(128,0,128)],
    ["RebeccaPurple",(102,51,153)],
    ["Red",(255,0,0)],
    ["RosyBrown",(188,143,143)],
    ["RoyalBlue",(65,105,225)],
    ["SaddleBrown",(139,69,19)],
    ["Salmon",(250,128,114)],
    ["SandyBrown",(244,164,96)],
    ["SeaGreen",(46,139,87)],
    ["SeaShell",(255,245,238)],
    ["Sienna",(160,82,45)],
    ["Silver",(192,192,192)],
    ["SkyBlue",(135,206,235)],
    ["SlateBlue",(106,90,205)],
    ["SlateGray",(112,128,144)],
    ["SlateGrey",(112,128,144)],
    ["Snow",(255,250,250)],
    ["SpringGreen",(0,255,127)],
    ["SteelBlue",(70,130,180)],
    ["Tan",(210,180,140)],
    ["Teal",(0,128,128)],
    ["Thistle",(216,191,216)],
    ["Tomato",(255,99,71)],
    ["Turquoise",(64,224,208)],
    ["Violet",(238,130,238)],
    ["Wheat",(245,222,179)],
    ["White",(255,255,255)],
    ["WhiteSmoke",(245,245,245)],
    ["Yellow",(255,255,0)],
    ["YellowGreen",(154,205,50)],
    ]

label_str = ["","aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"
];

labels_of_interest = {
    'none': (0, 'Background'),
    'bicycle': (1, 'bicycle'),
    'bus': (2, 'bus'),
    'car': (3, 'car'),
    'motorbike': (4, 'motorbike'),
    'person': (5, 'person'),
    'train': (6, 'train'),
    'stop' : (7, 'stop'),
    'pedestrianCrossing' : (8, 'pedestrianCrossing'),
    'signalAhead' : (9, 'signalAhead'),
    'speedLimit15' : (10, 'speedLimit'),
    'speedLimit25' : (10, 'speedLimit'),
    'speedLimit30' : (10, 'speedLimit'),
    'speedLimit35' : (10, 'speedLimit'),
    'speedLimit40' : (10, 'speedLimit'),
    'speedLimit45' : (10, 'speedLimit'),
    'speedLimit50' : (10, 'speedLimit'),
    'speedLimit55' : (10, 'speedLimit'),
    'speedLimit65' : (10, 'speedLimit'),
    'speedLimitUrdbl' : (10, 'speedLimit'),
}
# label id map
label_to_id = {label_text:group[0]
                for label_text, group in labels_of_interest.iteritems()}
id_to_label = {int(group[0]):group[1]
                for label_text, group in labels_of_interest.iteritems()}
#id_to_label = label_str
# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


# =========================================================================== #
# OpenCV show boxes
# =========================================================================== #
def plt_bboxes(img, classes, scores, bboxes, id_to_label_box, figsize=(10,10),
               Threshold =None, linewidth=1.5, bboxRecord=None, imagename=None):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    #fig = plt.figure(1,figsize=figsize)

    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        if Threshold is not None:
            if scores[i] < Threshold:
                continue
        cls_id = int(classes[i])
        if cls_id >= 0:# and cls_id != 7 and cls_id != 76:
            score = scores[i]
            color = colors_custom[(cls_id%len(colors_custom))][1][::-1]
            # the higher the score, the clear the color
            # color = [score*c+(1-score)*255 for c in color]
            #color = colors_custom[cls_id]
            class_name = id_to_label_box[cls_id]['name'];
            # add detection result to bboxRecord
            if (bboxRecord is not None) and (imagename is not None):
                bboxRecord.append(imagename,
                                class_name,
                                [bboxes[i,1], # xmin
                                bboxes[i,0],  # ymin
                                bboxes[i,3],  # xmax
                                bboxes[i,2]]) # ymax
            # absolute value of bboxes
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),color, 2)
            #rect = plt.Rectangle((xmin, ymin), xmax - xmin,
            #                     ymax - ymin, fill=False,
            #                     edgecolor=colors[cls_id],
            #                     linewidth=linewidth)
            #plt.gca().add_patch(rect)
            cv2.putText(img, '{:s} | {:.3f}'.format(class_name, score), (xmin, ymin-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
            #plt.gca().text(xmin, ymin - 2,
            #               '{:s} | {:.3f}'.format(class_name, score),
            #               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
            #               fontsize=12, color='white')
        # varify bbox records
        # visualize_bbox_record(imagename, bboxRecord.get_by_image_name(imagename))
    return img

def draw_mask(img, classes, scores, bboxes, id_to_label_box, figsize=(10,10),
               Threshold =None, linewidth=1.5, bboxRecord=None, imagename=None):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    #fig = plt.figure(1,figsize=figsize)

    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        if Threshold is not None:
            if scores[i] < Threshold:
                continue
        cls_id = int(classes[i])
        if cls_id >= 0:# and cls_id != 7 and cls_id != 76:
            score = scores[i]
            color = colors_custom[(cls_id%len(colors_custom))][1][::-1]
            # the higher the score, the clear the color
            # color = [score*c+(1-score)*255 for c in color]
            #color = colors_custom[cls_id]
            class_name = id_to_label_box[cls_id]['name'];
            # add detection result to bboxRecord
            if (bboxRecord is not None) and (imagename is not None):
                bboxRecord.append(imagename,
                                class_name,
                                [bboxes[i,1], # xmin
                                bboxes[i,0],  # ymin
                                bboxes[i,3],  # xmax
                                bboxes[i,2]]) # ymax
            # absolute value of bboxes
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),color, -1)
