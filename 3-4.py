names = ['yzq','zy','yhb','zhj']
print(names)
print(names[3]+"无法过来")
names[3] = 'zdy'
print(names)
names.append('wz')
names.append('yfl')
names.append('hqc')
print("我找到了更大的桌子并且邀请了" + str(names))
names.insert(0,'wmy')
print("我找到了更大的桌子并且邀请了" + str(len(names)) + str(names))
for i in range(6):
    names.pop()
print(names)
for i in range(2):
    del names[0]
print(names)