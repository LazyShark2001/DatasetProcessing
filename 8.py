# animal = "cate"
# name = "kiki"
# a = ["qiqi"]
# def pet(name,animal="cat"):
#     print("my animal is "+animal+",he name is "+name)
#     return [name,animal]
# a = a+pet(name, animal=animal)
# print(a)

#
# a = ["123","34"]
# def b(c):
#     c[0] = "1"
# b(a)
# print(a)
#
#
# def profile(first,last,**userinfo):
#     userinfo['first_name'] = first
#     userinfo['last_name'] = last
#     return userinfo
# user_info = profile('易','志琦',location = '湖北省武汉市',time = '23年6月19')
# print(user_info)


def a(name , age ,*hh ,it = "dog"):
    print("he is "+it + " "+name+" age is "+age+" "+str(hh))
a("cat","15","ssad","sad")