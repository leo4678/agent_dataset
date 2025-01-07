import xml.etree.ElementTree as ET
import random
import time
import subprocess
import hashlib

def get_file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

subprocess.run(f"rm ./output/*", shell=True)
subprocess.run(f"rm output.json", shell=True)

file_count = 0
home_activity = subprocess.run(f"adb shell dumpsys activity activities | grep mResumedActivity",
                        shell=True, text=True, capture_output=True)
subprocess.run(f"adb shell screencap -p /sdcard/0.png", shell=True)
subprocess.run(f"adb pull /sdcard/0.png ./output/", shell=True)
subprocess.run(f"adb shell uiautomator dump --verbose /sdcard/0.xml", shell=True)
subprocess.run(f"adb pull /sdcard/0.xml ./output/", shell=True)

repetitions = 0
interface_identifier = 1
coordinates = []
control_names = []

for _ in range(50):
    file_count += 1
    img_name = f"{file_count}.png"
    xml_name = f"{file_count}.xml"
    tree = ET.parse(f'output/{file_count-1}.xml')
    root = tree.getroot()
    controls = root.findall('.//node')
    if controls:
        random_control = controls[random.randint(0, len(controls) - 1)]
        bounds = random_control.attrib['bounds']
        bounds_pairs = bounds.split('][')
        x1, y1 = map(int, bounds_pairs[0].replace('[', '').split(','))
        x2, y2 = map(int, bounds_pairs[1].replace(']', '').split(','))
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        coordinates.append((x, y))
        control_name = random_control.attrib.get('text', '')  # 获取控件名称，如果不存在则返回空字符串
        if control_name == '':
            control_names.append(' ')  # 如果控件名称为空，则添加空字符串作为占位符
        else:
            control_names.append(control_name)
        #print(control_names)
        subprocess.run(f"adb shell input tap {x} {y}", shell=True)
        time.sleep(0.5)
        subprocess.run(f"adb shell screencap -p /sdcard/{img_name}", shell=True)
        subprocess.run(f"adb pull /sdcard/{img_name} ./output/", shell=True)
        result = subprocess.run(f"adb shell uiautomator dump --verbose /sdcard/{xml_name}", shell=True,
                                text=True, capture_output=True)
        if result.returncode != 0:
            interface_identifier += 1
            continue
        subprocess.run(f"adb pull /sdcard/{xml_name} ./output/", shell=True)
        old_hash = get_file_md5(f'output/{file_count-1}.xml')
        new_hash = get_file_md5(f'output/{xml_name}')
        if old_hash != new_hash:
            print("进入新界面，继续点击")
            repetitions = 0
            interface_identifier += 1
            print(interface_identifier)
        else:
            print("界面未更新，继续在当前界面操作")
            repetitions += 1
            print(repetitions)
            if repetitions == 3:
                print("界面超过3次未更新，返回上个界面")
                subprocess.run(f"adb shell input keyevent 4", shell=True)
                interface_identifier -= 1
                print(interface_identifier)
    else:
        print("当前界面未获取到控件，返回上个界面")
        subprocess.run(f"adb shell input keyevent 4", shell=True)
        interface_identifier -= 1
        print(interface_identifier)
    if interface_identifier > 5:
        print(interface_identifier)
        print("返回首页")
        for _ in range(4):
            current_activity = subprocess.run(f"adb shell dumpsys activity activities | grep mResumedActivity",
                                              shell=True, text=True, capture_output=True)
            print(home_activity.stdout)
            print(current_activity.stdout)
            if current_activity.stdout != home_activity.stdout:
                subprocess.run(f"adb shell input keyevent 4", shell=True)
                time.sleep(0.1)
        interface_identifier = 1
