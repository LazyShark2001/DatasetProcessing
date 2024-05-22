# active = True
# while active:
#     a = input('请输入您的年龄:')
#     if a=="quit":break
#     else:i = int(a)
#     if i<3:print("门票免费")
#     elif i<=12:print("10")
#     elif i>12:print("15")
#
#
# import random
#
# lucks = {}
# flag = True
# while flag:
#     name = input("请输入您的名字：")
#     luck = input("请输入您的幸运值：")
#     a = random.randint(1, 100)
#     if a<=int(luck):
#         b = input("您确实很幸运，我们记下了你的幸运值，您还要再生成一个记录吗?")
#         lucks[name] = luck
#     else:
#         b = input("幸运女神这次没有眷顾您，您还要再试一次吗")
#     if b.lower() == "yes":
#         continue
#     else:break
# print("已退出，谢谢您的助力，结果如下：")
# print(lucks)

# import  random
# a = 0
# while a<20:
#     print(random.randint(1,100))
#     a = a+1
