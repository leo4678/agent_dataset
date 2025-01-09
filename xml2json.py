import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
import json

from chick import coordinates, control_names, interface_identifier, bounds_dict

path="/Users/qilin/zxd/DiffusersExample/output"
file_count = 1
def traverse_xml(element):
    global file_count
    for node in element.findall('node'):
        node_data={
            'index': node.attrib['index'],
            'text': node.attrib['text'],
            'resource-id': node.attrib['resource-id'],
            'class': node.attrib['class'],
            'package': node.attrib['package'],
            'content-desc': node.attrib['content-desc'],
            'checkable': node.attrib['checkable'],
            'checked': node.attrib['checked'],
            'clickable': node.attrib['clickable'],
            'enabled': node.attrib['enabled'],
            'focusable': node.attrib['focusable'],
            'focused': node.attrib['focused'],
            'scrollable': node.attrib['scrollable'],
            'long-clickable': node.attrib['long-clickable'],
            'password': node.attrib['password'],
            'selected': node.attrib['selected'],
            'bounds': node.attrib['bounds'],
            'xpos': node.attrib['xpos']
        }
        #if node_data["text"] or node_data["content-desc"]:
            #if node_data["clickable"] == "true":
        if node_data["class"] == "android.widget.ImageView" or node_data["class"] == "android.widget.TextView":
            bound = node.attrib['bounds']
            bound_pairs = bound.split('][')
            x10, y10 = map(int, bound_pairs[0].replace('[', '').split(','))
            x11, y11 = map(int, bound_pairs[1].replace(']', '').split(','))
            center_x = (x10 + x11) // 2
            center_y = (y10 + y11) // 2
            node_data["center_point"] = [center_x, center_y]
            node_data["marked_image"] = f'{file_count}.png'

            elements_data.append(node_data)
            #print(elements_data)
            file_count += 1

    for child in element:
        traverse_xml(child)

#def print_xml_tree(node, depth=0):
#    result=' '*depth*2 + f'Node:{node.tag}, Text:{node.attrib}'
#    for child in node:
#        result+=print_xml_tree(child, depth+1)
#    return result

def print_xml_tree2(node):
    node_data={
        'tag': node.tag,
        'attrib': node.attrib,
        'children': []
    }
    for child in node:
        node_data['children'].append(print_xml_tree2(child))
    return node_data

data=[]
for filename in os.listdir(path):
    folder_path = os.path.join(path, os.path.splitext(filename)[0])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if filename.endswith(".xml"):
        xmlfile_path=os.path.join(path, filename)

        name=xmlfile_path.split(".")[0]
        imgfile_name=name+".png"
        image=Image.open(imgfile_name)
        width, height=image.size

        tree=ET.parse(xmlfile_path)
        root=tree.getroot()

        dic={}
        elements_data=[]
        action_data={
            "action_type": "CLICK"
        }
        action_details_data={}
        match = re.search(r'(\d+)\.xml', filename)
        if int(match.group(1)) > 0:
            action_details_data["sequence_num"]=interface_identifier
            action_details_data["clicked_element_name"]=control_names[int(match.group(1))-1]
            action_details_data["clicked_element_bbox"]=bounds_dict[int(match.group(1))-1]
            action_details_data["clicked_point"]=coordinates[int(match.group(1))-1]

        traverse_xml(root)
        action_data["action_details"]=action_details_data
        dic["image_url"]=os.path.basename(imgfile_name)
        dic["image_size"]=[width, height]
        #dic["xml_etree"]=print_xml_tree2(root)
        dic["action_related"]=action_data
        dic["elements"]=elements_data
        data.append(dic)

        file_count = 1
        for elements in elements_data:
            bounds = elements['bounds']
            bounds_pairs = bounds.split('][')
            x0, y0 = map(int, bounds_pairs[0].replace('[', '').split(','))
            x1, y1 = map(int, bounds_pairs[1].replace(']', '').split(','))
            coords = [int(x0), int(y0), int(x1), int(y1)]
            x0, y0, x1, y1 = coords
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            img=Image.open(imgfile_name)
            draw = ImageDraw.Draw(img)
            draw.rectangle([x0, y0, x1, y1], outline='red', width=5)
            img.save(f'{folder_path}/{file_count}.png')
            file_count += 1

with open("output.json", "w") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)