import re
def get_ind_name(strings):
#     识别两种合同书写方式 第一条 或者 一
    pattern_d = re.compile('(第[一二三四五六七八九十]{1,2}(部分|条))')
    pattern_t = re.compile('([一二三四五六七八九十]{1,2}[、 ])')
    partname = {}
    if re.findall(pattern_d, strings) and re.findall(pattern_t, strings):
#         这是两种情况都存在的情况 此时 那个正则识别的多用哪个
        if len(re.findall(pattern_d, strings)) >= len(re.findall(pattern_t, strings)):
#         找到第一条描述的字符 放进list 然后找到字符所在的位置 将位置存储 然后遍历两个list 对合同进行拆分 之后大致都是这个流程
            patter = [c[0].replace('、', '').replace(' ', '') for c in re.findall(pattern_d, strings)]
            pattern = []
            for i in patter:
                if i not in pattern:
                    pattern.append(i)
            index = [strings.index(i) for i in pattern]
            index.append(len(strings) - 1)
            former = strings[:index[0]].split('\n')
            for key, value in zip(pattern, range(len(index) - 1)):
                partname[key] = strings[index[value]:index[value + 1]]
            return partname, former
        else:
            patter = [c.replace('、', '').replace(' ', '') for c in re.findall(pattern_t, strings)]
            pattern = []
            for i in patter:
                if i not in pattern:
                    pattern.append(i)
            index = [strings.index(i) for i in pattern]
            index.append(len(strings) - 1)

            former = strings[:index[0]].split('\n')
            for key, value in zip(pattern, range(len(index) - 1)):
                partname['第' + key.replace('、', '') + '条'] = strings[index[value]:index[value + 1]]
            return partname, former
    elif re.findall(pattern_d, strings):
        patter = [c[0].replace('、', '').replace(' ', '') for c in re.findall(pattern_d, strings)]
        pattern = []
        for i in patter:
            if i not in pattern:
                pattern.append(i)
        index = [strings.index(i) for i in pattern]
        index.append(len(strings) - 1)
        former = strings[:index[0]].split('\n')
        for key, value in zip(pattern, range(len(index) - 1)):
            partname[key] = strings[index[value]:index[value + 1]]
        return partname, former
    elif re.findall(pattern_t, strings):
        patter = [c.replace('、', '').replace(' ', '') for c in re.findall(pattern_t, strings)]
        pattern = []
        for i in patter:
            if i not in pattern:
                pattern.append(i)
        index = [strings.index(i) for i in pattern]
        index.append(len(strings) - 1)
        former = strings[:index[0]].split('\n')
        for key, value in zip(pattern, range(len(index) - 1)):
            partname['第' + key.replace('、', '') + '条'] = strings[index[value]:index[value + 1]]
        return partname, former
    else:
        return False, False
