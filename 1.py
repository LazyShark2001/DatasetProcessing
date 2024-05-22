# names = {'yzq' :'1','zy':'2','yhb' : '3','zhj':'4'}
# ssad = ['asd','asd']
# print(names.pop('yz',None))
# print(names)
# sada = 'asdadsas'
# print(sada.index('d'))
#
# names = ['small-vehicle', 'large-vehicle', 'plane', 'storage-tank', 'ship', 'harbor', 'ground-track-field', 'soccer-ball-field', 'tennis-court', 'swimming-pool', 'baseball-diamond', 'roundabout', 'basketball-court', 'bridge', 'helicopter']
# its = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
# sss = { k:v for k,v in zip(its,names) }
# a1,a2 = zip(*zip(its,names))
# print(str(a1)+"\n"+str(a2))
# c = zip(its,names)
# print(c)
# map()


# players = [1,3,5,9,3,1,5,1,2,3,'a']
# A = players
# A[0] = 'wos'
#
# print(str(players[0])+players[-1])
# print(A.index())
#


from difflib import SequenceMatcher

cars = ['audi','bmw','subaru','toyota','maybach']
for car in cars:
    i = SequenceMatcher(a=car,b=' b mw ').ratio()
    print(i)
    if i > 0.6:
        print(car.upper())
    else:
        print(car.title())



print('1'==1)