import re
import xml.etree.ElementTree as ET
from PIL import Image
import os
import json
from chick import coordinates, control_names, interface_identifier

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
        if node_data["text"] or node_data["content-desc"]:
            if node_data["checkable"] == "true" or node_data["enabled"] == "true" or \
                node_data["clickable"] == "true" or node_data["long-clickable"] == "true":
                elements_data.append(node_data)

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
            action_details_data["clicked_element_bbox"]=coordinates[int(match.group(1))-1]
            action_details_data["clicked_point"]=""

        traverse_xml(root)
        action_data["action_details"]=action_details_data
        dic["image_url"]=os.path.basename(imgfile_name)
        dic["image_size"]=[width, height]
        #dic["xml_etree"]=print_xml_tree2(root)
        dic["action_related"]=action_data
        dic["elements"]=elements_data
        data.append(dic)

with open("output.json", "w") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)