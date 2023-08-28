import re

DEEP_RECURSION = '3'
last = 12
prev = 10
Q = 3

data = ['raz', 'dva', 'tri']
for g in range(1, (int(DEEP_RECURSION)) + 1):

    last_l = [last, prev] if Q != 1 else last
    prev_p = [last - Q, prev - Q] if Q != 1 else prev
    # print(last_l,prev_p)
    if g != 1:
        last -= 1
        prev -= 1
        last_l = [last, prev] if Q != 1 else last
        prev_p = [last-Q, prev-Q] if Q != 1 else prev


    print(last_l,prev_p)
    print('__________')
    for i in data:
        print(i,1)



        # print(last-g)
        # print(prev-g)
        # zx = f'_all_'
        # print(re.match(r"\__all__\d{1}",zx))

        # if re.fullmatch(r"\_all_\d*",zx) is not None:
        #     pass
# #
# #
#     print('zzzzzzz')
#     print(g)
# d = '__all__f'
# m = re.sub(r"\__all__\d*", '__all__',d)
# m = re.fullmatch(r"\__all__\d*",d)
# print(m)

req = ['raz', 'dva', 'tri']
for index, enumerate in enumerate(req):
    print(')))))))))')
    print(index,enumerate)