import xml.etree.ElementTree as ET
from PIL import Image
import os
import json

path="/Users/qilin/zxd/DiffusersExample/output"

def traverse_xml(element):
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
        elements_data.append(node_data)

        action_details_data["sequence_num"]=""
        action_details_data["clicked_element_name"]=node.attrib['text']
        action_details_data["clicked_element_bbox"]=node.attrib['bounds']
        action_details_data["clicked_point"]=""

    for child in element:
        traverse_xml(child)

def print_xml_tree(node, depth=0):
    result=' '*depth*2 + f'Node:{node.tag}, Text:{node.attrib}'
    for child in node:
        result+=print_xml_tree(child, depth+1)
    return result

data=[]
for filename in os.listdir(path):
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
        traverse_xml(root)
        action_data["action_details"]=action_details_data
        dic["image_url"]=os.path.basename(imgfile_name)
        dic["image_size"]=[width, height]
        dic["xml_etree"]=print_xml_tree(root)
        dic["action_related"]=action_data
        dic["elements"]=elements_data
        data.append(dic)

with open("output.json", "w") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)