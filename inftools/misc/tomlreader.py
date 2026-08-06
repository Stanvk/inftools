def infretis_data_reader(input):
    ens_dic = {}
    num_ens = 0
    with open(input, 'r') as read:
        # the header is either a single '#intf:' line or the older three-line
        # '# ===' block, so skip comments instead of counting header lines
        for line in read:
            if line.startswith('#'):
                continue
            rip = line.rstrip().split()[3:]
            pinfo = line.rstrip().split()[:3]
            if not num_ens:
                # each ensemble has a weight and a high acceptance weight
                num_ens = len(rip)//2
                ens_dic = {i : {'op': [], 'haw': [], 'w': []}
                           for i in range(num_ens)}
            for idx, (w, haw) in enumerate(zip(rip[:num_ens], rip[num_ens:])):
                if '----' in (w, haw):
                    continue
                ens_dic[idx]['op'].append(float(pinfo[2]))
                ens_dic[idx]['w'].append(float(w))
                ens_dic[idx]['haw'].append(float(haw))
    return ens_dic

