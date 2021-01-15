import sys
import json

m_old = {}
m_new = {}

with open(sys.argv[1], 'r') as inp:
    for line in inp:
        if len(line) < 5:
            continue

        j = json.loads(line)
        if 'Ogrn' not in j:
            print('NO OGRN', line)
        o = j['Ogrn']

        is_old = line.startswith('\t')
        dic_to_add = m_old if is_old else m_new
        dic_to_add[o] = dic_to_add.get(o, {})
        dic_to_add[o]['m1'] = j['Markers'] if 'Markers' in j else 0
        dic_to_add[o]['m2'] = j['Markers2'] if 'Markers2' in j else 0

with open('ogrns_old_uniq', 'w') as out:
    for o in set(m_old.keys()) - set(m_new.keys()):
        out.write(f'{o}\n')

with open('ogrns_new_uniq', 'w') as out:
    for o in set(m_new.keys()) - set(m_old.keys()):
        out.write(f'{o}\n')

with open('markers_diff', 'w') as out:
    out.write('old_diff\tnew_diff\n')
    for o in set(m_old.keys()).intersection(set(m_new.keys())):
        m1_diff = bin(m_old[o]["m1"] ^ m_new[o]["m1"])[2:]
        m1_diff = list(map(lambda a: a[1], filter(lambda x: x[0] == '1', zip(m1_diff, range(len(m1_diff))))))
        m2_diff = bin(m_old[o]["m2"] ^ m_new[o]["m2"])[2:]
        m2_diff = list(map(lambda a: a[1], filter(
            lambda x: x[0] == '1', zip(m1_diff, range(len(m2_diff))))))
        out.write(f'{m1_diff}\t{m2_diff}\n')
