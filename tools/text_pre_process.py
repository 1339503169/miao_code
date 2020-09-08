import re
def get_ind_name(strings):
    pattern_d = re.compile('(第[一二三四五六七八九十]{1,2}(部分|条))')
    pattern_t = re.compile('([一二三四五六七八九十]{1,2}[、 ])')
    partname = {}
    if re.findall(pattern_d, strings) and re.findall(pattern_t, strings):
        if len(re.findall(pattern_d, strings)) >= len(re.findall(pattern_t, strings)):
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