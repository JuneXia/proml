import numpy as np

def func():
    ret = -1
    s = input()
    try:
        numlist = s.split(',')
        numlist = np.array(numlist, dtype=np.int32).tolist()
        if len(set(numlist)) != 3 or min(numlist) <= 0:
            return -1

        if 2 in numlist:
            numlist.append(5)
        elif 5 in numlist:
            numlist.append(2)
        if 6 in numlist:
            numlist.append(9)
        elif 9 in numlist:
            numlist.append(6)

        max_num = max(numlist)

        numlist.sort()
        pre_i = 0
        end_i = 0
        num_size = len(numlist)
        while len(numlist) < 9:
            end_i += 1
            if pre_i == end_i:
                continue

            numlist.append(numlist[pre_i] * 10 + numlist[end_i])
            if end_i + 1 == num_size:
                pre_i += 1
                end_i = 0


        return numlist[max_num]
    except:
        return -1


if __name__ == '__main__':
    print(func())