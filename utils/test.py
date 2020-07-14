import xml.etree.ElementTree as ET
tree = ET.parse('new_file.xml')
root = tree.getroot()
new_file_name = 'file_name.xml'
ratio = 1280 / 2000

for child in root:
    if child.tag == 'object':
        child[4][0].text = str(round(int(child[4][0].text) * ratio))
        child[4][1].text = str(round(int(child[4][1].text) * ratio))
        child[4][2].text = str(round(int(child[4][2].text) * ratio))
        child[4][3].text = str(round(int(child[4][3].text) * ratio))

tree.write(new_file_name)