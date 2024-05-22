# for i in range(21):
#     print(i)

# s = [value for value in range(1,1000001)]
# print(min(s))
# print(max(s))
# print(sum(s))

# q = [i**3 for i in range(1,11)]
# for i in q:
#     print(i)
# # 输出菜单
# print('菜单：')
# for k, v in menu.items():
#     print(k, '-', v, '元')
#



# # 输入点餐
# order = input('请输入要点的菜品名及份数，用空格分隔：\n')
#
# # 处理点餐
# total_price = 0
# order_list = order.split(' ')
# for o in order_list:
#     dish, count = o.split(',')
#     total_price += menu[dish] * int(count)
#
# # 输出点餐信息
# print('点餐信息：')
# print('菜品', '份数', '单价', '总价')
# for o in order_list:
#     dish, count = o.split(',')
#     print(dish, count, menu[dish], menu[dish] * int(count))
# print('总价：', total_price, '元')