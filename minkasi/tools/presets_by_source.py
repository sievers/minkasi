import numpy


def get_src_presets(name):

    f3c = name[0:3]

    if f3c == "MOO":
        myadj, coords, nsrc, JY2K, fitp, aa, bb, rot = get_madcows_def(name)
    else:
        myadj, coords, nsrc, JY2K, fitp, aa, bb, rot = get_m2team_def(name)

    return myadj, coords, nsrc, JY2K, fitp, aa, bb, rot


def get_madcows_def(name):

    gpdir = "/home/scratch/cromero/mustang/MUSTANG2/Reductions/" + name + "/Minkasi/"
    fitp = None
    aa = 1.0
    bb = 1.0
    rot = 0.0

    if name == "MOO_0105":
        # myadj='TS_-20_def_ninkasi_PdoCals'
        # myadj='TS_EaCMS_51_7_Aug_2019'
        myadj = "TS_EaCMS0f0_51_8_Jan_2020"
        # centfile = 'moo0105_1src_gaussp.npy'
        # centfile = gpdir+name+'_gaussp_5Mar2019_minchi_all_v0.npy'
        centfile = gpdir + name + "_gaussp_9Jan2020_minchi_all_v0.npy"
        fitp = numpy.load(centfile)
        coords = numpy.zeros([fitp.shape[0] // 4, 2])
        coords[:, 0] = fitp[0::4]
        coords[:, 1] = fitp[1::4]
        coords = coords * 180 / numpy.pi
        clus_cent = numpy.asarray(
            [coords[0, 0], coords[0, 1], 0.004, 0.0], dtype="double"
        )
        ptsrc_cent = numpy.asarray(
            [coords[1, 0], coords[1, 1], 0.0004, 0.0], dtype="double"
        )
        JY2K = 0.791007611858
        nsrc = 1
    #######################################################################################
    if name == "MOO_0135":
        myadj = "TS_v0_0f02_51_Jan26"
        clus_cent = numpy.asarray([23.76198, 32.13258, 0.004, 0.0], dtype="double")
        ptsrc_cent = numpy.asarray([0.0, 0.0, 0.000035, 0.0], dtype="double")
        nsrc = 0
        JY2K = 0.702884305514
    #######################################################################################
    if name == "MOO_1014":
        myadj = "TS_-20_def_ninkasi_PdoCals"
        clus_cent = numpy.asarray([153.52928, 0.63918297, 0.0004, 0.0], dtype="double")
        ptsrc_cent = numpy.asarray([0.0, 0.0, 0.000035, 0.0], dtype="double")
        nsrc = 0
        JY2K = 0.677688893964
    #######################################################################################
    if name == "MOO_1046":
        # myadj='TS_EaCMS_51_7_Aug_2019'
        myadj = "TS_EaCMS0f0_51_17_Feb_2020"
        centfile = gpdir + name + "_gaussp_5Mar2019_minchi_all_v0.npy"
        fitp = numpy.load(centfile)
        coords = numpy.zeros([fitp.shape[0] // 4, 2])
        coords[:, 0] = fitp[0::4]  # 161.71933206
        coords[:, 1] = fitp[1::4]  # 27.96784252
        coords = coords * 180 / numpy.pi
        print(coords)
        # import pdb;pdb.set_trace()
        clus_cent = numpy.asarray(
            [coords[0, 0], coords[0, 1], fitp[2] * 53.0, fitp[3]], dtype="double"
        )
        # clus_cent=numpy.asarray([161.7190594,27.96636437,fitp[2]*53.0,fitp[3]],dtype='double')
        ptsrc_cent = numpy.asarray(
            [coords[1, 0], coords[1, 1], fitp[6] * 180.0 / numpy.pi, fitp[7]],
            dtype="double",
        )
        nsrc = 1
        # clus_cent = numpy.asarray([161.7185,27.966637,0.004,0.0],dtype='double')
        # ptsrc_cent = numpy.asarray([161.75692,27.945782,0.000035,0.0],dtype='double')
        JY2K = 0.727984471080
    #######################################################################################
    if name == "MOO_1059":
        # myadj='TS_v0_51_31_Jan_2019'
        myadj = "TS_EaCMS0f0_51_9_Jan_2020"
        centfile = gpdir + name + "_gaussp_5Mar2019_minchi_all_v0.npy"
        fitp = numpy.load(centfile)
        coords = numpy.zeros([fitp.shape[0] // 4, 2])
        coords[:, 0] = fitp[0::4]
        coords[:, 1] = fitp[1::4]
        coords = coords * 180 / numpy.pi
        clus_cent = numpy.asarray(
            [coords[0, 0], coords[0, 1], fitp[2] * 53.0, fitp[3]], dtype="double"
        )
        ptsrc_cent = numpy.asarray(
            [coords[1, 0], coords[1, 1], fitp[6] * 180.0 / numpy.pi, fitp[7]],
            dtype="double",
        )
        nsrc = 1
        # clus_cent = numpy.asarray([164.9666032,54.91955515,0.004,0.0],dtype='double')
        # ptsrc_cent = numpy.asarray([165.0575529,54.92889663,0.0004,0.0],dtype='double')
        JY2K = 0.766842216585
    #######################################################################################
    if name == "MOO_1110":
        # myadj='TS_v0_0f02_51_Jan26'
        myadj = "TS_EaCMS_51_8_Aug_2019"
        centfile = gpdir + name + "_gaussp_5Mar2019_minchi_all_v0.npy"
        fitp = numpy.load(centfile)
        coords = numpy.zeros([fitp.shape[0] // 4, 2])
        coords[:, 0] = fitp[0::4]
        coords[:, 1] = fitp[1::4]
        coords = coords * 180 / numpy.pi
        clus_cent = numpy.asarray(
            [coords[0, 0], coords[0, 1], fitp[2] * 53.0, fitp[3]], dtype="double"
        )
        ptsrc_cent = numpy.asarray(
            [coords[1, 0], coords[1, 1], fitp[6] * 180.0 / numpy.pi, fitp[7]],
            dtype="double",
        )
        # clus_cent = numpy.asarray([167.73388,68.641604,0.004,0.0],dtype='double')
        # ptsrc_cent = numpy.asarray([167.8127,68.648289,0.0004,0.0],dtype='double')
        nsrc = 1
        JY2K = 0.845818487150
    #######################################################################################
    if name == "MOO_1142":
        myadj = "TS_EaCMS0f0_51_17_Feb_2020"
        # myadj='TS_EaCMS0f0_51_9_Jan_2020'
        # clus_cent = numpy.asarray([175.68962,15.454278,0.004,0.0],dtype='double')
        # ptsrc_cent = numpy.asarray([175.698466,15.45273403,0.003,0.0],dtype='double')
        centfile = gpdir + "MOO_1142_gaussp_5Mar2019_minchi_all_v0.npy"
        fitp = numpy.load(centfile)
        coords = numpy.zeros([fitp.shape[0] // 4, 2])
        coords[:, 0] = fitp[0::4]
        coords[:, 1] = fitp[1::4]
        coords = coords * 180 / numpy.pi
        clus_cent = numpy.asarray(
            [coords[0, 0], coords[0, 1], fitp[2] * 53.0, fitp[3]], dtype="double"
        )
        ptsrc_cent = numpy.asarray(
            [coords[1, 0], coords[1, 1], fitp[6] * 180.0 / numpy.pi, fitp[7]],
            dtype="double",
        )
        nsrc = 1
        JY2K = 0.999031659597
    #######################################################################################
    if name == "MOO_1329":
        myadj = "TS_EaCMS0f0_51_31_Jan_2020"
        clus_cent = numpy.asarray([202.46314, 56.79894, 0.002, -0.0005], dtype="double")
        # ptsrc_cent = numpy.asarray([202.46104,56.787382,0.0004,0.0005,
        #                         202.4655,56.782411,0.0004,0.0003],dtype='double')
        centfile = gpdir + name + "_gaussp_1Feb2020_minchi_all_v0.npy"
        fitp = numpy.load(centfile)
        coords = numpy.zeros([fitp.shape[0] // 4, 2])
        coords[:, 0] = fitp[0::4]
        coords[:, 1] = fitp[1::4]
        coords = coords * 180 / numpy.pi
        ### NOPE, I don't like its fits.
        coords[0, 0] = clus_cent[0]
        coords[0, 1] = clus_cent[1]
        nsrc = 2
        JY2K = 0.63618
    #######################################################################################
    if name == "MOO_1506":
        myadj = "TS_v0_51_4_Feb_2019"
        clus_cent = numpy.asarray([226.5951, 51.608676, 0.004, 0.0], dtype="double")
        ptsrc_cent = numpy.asarray([0.0, 0.0, 0.02, 0.0], dtype="double")
        coords = numpy.zeros([2, 2])
        coords[0, 0] = clus_cent[0]
        coords[0, 1] = clus_cent[1]
        coords[1, 0] = ptsrc_cent[0]
        coords[1, 1] = ptsrc_cent[1]
        nsrc = 1
        JY2K = 0.797261154974
    #######################################################################################

    return myadj, coords, nsrc, JY2K, fitp, aa, bb, rot


def get_m2team_def(name):

    outdir = "/home/scratch/cromero/mustang/MUSTANG2/Reductions/" + name + "/Minkasi/"
    npar_src = 4
    if name == "Zw3146":
        myadj = "TS_EaCMS_51_13_Aug_2019"  # The best ones still..
        fitp = numpy.load(
            outdir + "GaussPars/zwicky_7src_gaussp_20Jun2019_edit_from_3Feb2019.npy"
        )
        coords = numpy.zeros([fitp.shape[0] // npar_src, 2])
        coords[:, 0] = fitp[0::npar_src]
        coords[:, 1] = fitp[1::npar_src]
        coords = coords * 180 / numpy.pi
        nsrc = coords.shape[0] - 1
        JY2K = 0.757873406846
        aa = 1.0
        bb = 1.0
        rot = 0.0

    if name == "MS0735":
        myadj = "TS_EaCMS0f0_51_11_Jun_2020"  # The best ones still..
        # fitp=numpy.load(outdir+'GaussPars/zwicky_7src_gaussp_20Jun2019_edit_from_3Feb2019.npy')
        clus = [
            (7.0 + 41.0 / 60.0 + 45.0264 / 3600.0) * 15.0,
            74.0 + 14.0 / 60.0 + 30.974 / 3600.0,
            1e-4,
            -2e-4,
        ]
        clus[0] = clus[0] * numpy.pi / 180  # Now in radians!
        clus[1] = clus[1] * numpy.pi / 180  # Now in radians!
        agn = [
            (7.0 + 41.0 / 60.0 + 44.4185 / 3600.0) * 15.0,
            74.0 + 14.0 / 60.0 + 36.229 / 3600.0,
            1e-5,
            2e-4,
        ]
        agn[0] = agn[0] * numpy.pi / 180  # Now in radians!
        agn[1] = agn[1] * numpy.pi / 180  # Now in radians!
        clus.extend(agn)
        fitp = numpy.asarray(clus)
        coords = numpy.zeros([fitp.shape[0] // npar_src, 2])
        coords[:, 0] = fitp[0::npar_src]
        coords[:, 1] = fitp[1::npar_src]
        coords = coords * 180 / numpy.pi
        nsrc = coords.shape[0] - 1
        JY2K = 0.6823709672354434
        aa = 55.0 / 38.5  # By eye
        bb = 1.0
        # rot  = 1.40       # 1.396652389755478
        rot = 1.60  # M2 data seems a bit more rotated...

    return myadj, coords, nsrc, JY2K, fitp, aa, bb, rot


def get_bad_tods(name, ndo=False, odo=False):

    try:
        name = str(name).lower().strip("_")
    except TypeError:
        print("Error: name is not a string. TOD cuts will not be properly applied")
        return [], ""

    addtag = "svprods_"

    bad_215_01 = [
        "Signal_TOD-AGBT18B_215_01-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_02 = [
        "Signal_TOD-AGBT18B_215_02-s" + str(ss) + ".fits" for ss in [6, 8, 10, 17]
    ]  # Maybe 6,7 and 8 too?
    bad_215_03 = [
        "Signal_TOD-AGBT18B_215_03-s" + str(ss) + ".fits"
        for ss in [11, 13, 14, 15, 18, 20, 24, 25]
    ]  # Maybe 6,7 and 8 too?
    bad_215_04 = [
        "Signal_TOD-AGBT18B_215_04-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    # bad_215_04=['Signal_TOD-AGBT18B_215_04-s'+str(ss)+'.fits' for ss in range(40,120)] # Maybe 6,7 and 8 too?
    bad_215_05 = [
        "Signal_TOD-AGBT18B_215_05-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_06 = [
        "Signal_TOD-AGBT18B_215_06-s" + str(ss) + ".fits" for ss in [8, 9, 57, 65]
    ]  # Maybe 6,7 and 8 too?
    bad_215_07 = [
        "Signal_TOD-AGBT18B_215_07-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_08 = [
        "Signal_TOD-AGBT18B_215_08-s" + str(ss) + ".fits" for ss in [25, 26, 28, 85]
    ]  # Maybe 6,7 and 8 too?
    bad_215_09 = [
        "Signal_TOD-AGBT18B_215_09-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_10 = [
        "Signal_TOD-AGBT18B_215_10-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_11 = [
        "Signal_TOD-AGBT18B_215_11-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_12 = [
        "Signal_TOD-AGBT18B_215_12-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_215_13 = [
        "Signal_TOD-AGBT18B_215_13-s" + str(ss) + ".fits" for ss in [0]
    ]  # Maybe 6,7 and 8 too?
    bad_tod = bad_215_01
    bad_tod.extend(bad_215_02)
    bad_tod.extend(bad_215_03)
    bad_tod.extend(bad_215_04)
    bad_tod.extend(bad_215_05)
    bad_tod.extend(bad_215_06)
    bad_tod.extend(bad_215_07)
    bad_tod.extend(bad_215_08)
    bad_tod.extend(bad_215_09)
    bad_tod.extend(bad_215_10)
    bad_tod.extend(bad_215_11)
    bad_tod.extend(bad_215_12)
    bad_tod.extend(bad_215_13)

    bad_175_01 = [
        "Signal_TOD-AGBT18A_175_01-s" + str(ss) + ".fits" for ss in [9]
    ]  # Maybe 6,7 and 8 too?
    bad_175_03 = [
        "Signal_TOD-AGBT18A_175_03-s" + str(ss) + ".fits" for ss in [32, 36, 37, 38, 40]
    ]  # Mostly due to low elev.
    bad_175_04 = [
        "Signal_TOD-AGBT18A_175_04-s" + str(ss) + ".fits"
        for ss in [10, 11, 12, 28, 29, 32, 33, 34, 35, 44, 45, 46, 48, 49]
    ]
    bad_175_06 = [
        "Signal_TOD-AGBT18A_175_06-s" + str(ss) + ".fits"
        for ss in [38, 39, 40, 47, 48, 49, 50, 74, 75, 76, 77, 79, 80, 82, 83, 84]
    ]
    # bad_175_06=['Signal_TOD-AGBT18A_175_06-s'+str(ss)+'.fits' for ss in range(100)]
    bad_175_07 = [
        "Signal_TOD-AGBT18A_175_07-s" + str(ss) + ".fits"
        for ss in [12, 17, 19, 20, 22, 24, 25, 26, 27, 29, 30, 31]
    ]
    # bad_tod = bad_175_01   ;bad_tod.extend(bad_175_04);  bad_tod.extend(bad_175_06);  bad_tod.extend(bad_175_07)
    # bad_tod = bad_175_01   ;bad_tod.extend(bad_175_04);  bad_tod.extend(bad_175_07)

    bad_111_02 = [
        "Signal_TOD-AGBT18B_111_02-s" + str(ss) + ".fits"
        for ss in [7, 8, 16, 17, 18, 19, 21, 23, 24, 28, 30, 38, 39, 40]
    ]
    bad_111_03 = [
        "Signal_TOD-AGBT18B_111_03-s" + str(ss) + ".fits" for ss in [9, 10, 31]
    ]
    bad_111_04 = [
        "Signal_TOD-AGBT18B_111_04-s" + str(ss) + ".fits"
        for ss in [11, 12, 13, 14, 18, 19, 20, 21]
    ]
    bad_111_05 = ["Signal_TOD-AGBT18B_111_05-s" + str(ss) + ".fits" for ss in [14, 15]]
    # bad_tod = bad_111_02  ;bad_tod.extend(bad_111_03);  bad_tod.extend(bad_111_04);  bad_tod.extend(bad_111_05)

    bad_092_07 = ["Signal_TOD-AGBT19A_092_07-s" + str(ss) + ".fits" for ss in [25]]
    bad_tod.extend(bad_175_01)
    bad_tod.extend(bad_175_04)
    bad_tod.extend(bad_175_06)
    bad_tod.extend(bad_175_07)

    if name == "moo1046" or name == "mooj1046":
        # tod_files='/home/scratch/sdicker/AGBT18B_215/IDL_maps_current/moo1046/Signal_TOD-'+myproj+'.fits'
        # addtag="svprods_mcen_SDTODs_"
        if ndo:
            bad_tod = [
                "Signal_TOD-AGBT18B_215_10-s" + str(ss) + ".fits" for ss in range(30)
            ]  #
            addtag = "svprods_NDO_"
        if odo:
            bad_tod = [
                "Signal_TOD-AGBT18B_091_02-s" + str(ss) + ".fits" for ss in range(50)
            ]  #
            addtag = "svprods_ODO_"

    if name == "moo1142" or name == "mooj1142":
        bad_215_04 = [
            "Signal_TOD-AGBT18B_215_04-s" + str(ss) + ".fits"
            for ss in [52, 53, 54, 86, 87, 88, 89, 91, 92, 93]
        ]
        bad_019_02 = [
            "Signal_TOD-AGBT20A_019_02-s" + str(ss) + ".fits" for ss in [28, 43]
        ]
        bad_005_02 = [
            "Signal_TOD-AGBT23B_005_02-s" + str(ss) + ".fits" for ss in [17, 39]
        ]
        bad_005_08 = [
            "Signal_TOD-AGBT23B_005_08-s" + str(ss) + ".fits"
            for ss in [14, 16, 17, 18, 26, 27, 28, 29]
        ]
        bad_tod.extend(bad_215_04)
        bad_tod.extend(bad_019_02)
        bad_tod.extend(bad_005_02)
        bad_tod.extend(bad_005_08)
        if ndo:
            bad_tod = [
                "Signal_TOD-AGBT18B_215_04-s" + str(ss) + ".fits" for ss in range(100)
            ]  #
            addtag = "svprods_NDO_"
        if odo:
            bad_tod = [
                "Signal_TOD-AGBT18B_091_02-s" + str(ss) + ".fits" for ss in range(50)
            ]  #
            addtag = "svprods_ODO_"

    return bad_tod, addtag


def get_edges(maxbins, medbins, widebins, ultrawide, tenc):

    if maxbins:
        if tenc:
            edges = numpy.asarray(
                [
                    0,
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    105,
                    120,
                    135,
                    150,
                    165,
                    180,
                    210,
                    240,
                    270,
                    330,
                ],
                dtype="double",
            )
        else:
            edges = numpy.asarray(
                [
                    0,
                    5,
                    15,
                    25,
                    35,
                    45,
                    55,
                    65,
                    75,
                    85,
                    95,
                    105,
                    120,
                    135,
                    150,
                    165,
                    180,
                    210,
                    240,
                    270,
                    330,
                ],
                dtype="double",
            )
    elif medbins:
        if tenc:
            edges = numpy.asarray(
                [
                    0,
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    105,
                    120,
                    135,
                    150,
                    165,
                    180,
                    240,
                ],
                dtype="double",
            )
        else:
            edges = numpy.asarray(
                [
                    0,
                    5,
                    15,
                    25,
                    35,
                    45,
                    55,
                    65,
                    75,
                    85,
                    95,
                    105,
                    120,
                    135,
                    150,
                    165,
                    180,
                    240,
                ],
                dtype="double",
            )
    elif widebins:
        edges = numpy.asarray(
            [0, 10, 30, 50, 70, 90, 120, 150, 180, 240], dtype="double"
        )
    elif ultrawide:
        edges = numpy.asarray([0, 10, 30, 50, 70, 100, 130, 190, 280], dtype="double")
    else:
        if tenc:
            edges = numpy.asarray(
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 105, 120, 135, 150, 165, 180],
                dtype="double",
            )
        else:
            edges = numpy.asarray(
                [
                    0,
                    5,
                    15,
                    25,
                    35,
                    45,
                    55,
                    65,
                    75,
                    85,
                    95,
                    105,
                    120,
                    135,
                    150,
                    165,
                    180,
                ],
                dtype="double",
            )

    return edges
