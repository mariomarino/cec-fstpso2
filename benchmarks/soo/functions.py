from .dimensions import dimensions
from .definitions.standard import mishra01, quintic, michalewicz, shubert, alpine01, bohachevsky, ferretti, plateau, \
    xinsheyang02, vincent, griewank, ackley, rastrigin, schwefel, rosenbrock, nobile1, nobile2, nobile3, deceptive, \
    bukin06, cross_in_tray, drop_wave, egg_holder, holder_table, sphere, adjiman, alpine02, de_villiers_glasser02,\
    damavandi, xinsheyang03, whitley, salomon, stochastic, ripple01
from .definitions.perturbed import ackley_shifted, ackley_shrinked, alpine01_shrinked, alpine01_shifted,\
    rosenbrock_shrinked, rosenbrock_shifted, sphere_shrinked, sphere_shifted
# CEC 2022
from .definitions.CEC.TestSuite2022.CPP.cec2022 import cec22_f1, cec22_f2, cec22_f3, cec22_f4, cec22_f5, cec22_f6,\
    cec22_f7, cec22_f8, cec22_f9, cec22_f10, cec22_f11, cec22_f12
""""# CEC 2021 - Basic
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_basic, cec21_f2_basic, cec21_f3_basic, cec21_f4_basic,\
    cec21_f5_basic, cec21_f6_basic, cec21_f7_basic, cec21_f8_basic, cec21_f9_basic, cec21_f10_basic
# CEC 2021 - Bias
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_bias, cec21_f2_bias, cec21_f3_bias, cec21_f4_bias,\
    cec21_f5_bias, cec21_f6_bias, cec21_f7_bias, cec21_f8_bias, cec21_f9_bias, cec21_f10_bias
# CEC 2021 - Bias & Rotation
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_bias_rot, cec21_f2_bias_rot, cec21_f3_bias_rot,\
    cec21_f4_bias_rot, cec21_f5_bias_rot, cec21_f6_bias_rot, cec21_f7_bias_rot, cec21_f8_bias_rot, cec21_f9_bias_rot,\
    cec21_f10_bias_rot
# CEC 2021 - Bias & Shift
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_bias_shift, cec21_f2_bias_shift, cec21_f3_bias_shift,\
    cec21_f4_bias_shift, cec21_f5_bias_shift, cec21_f6_bias_shift, cec21_f7_bias_shift, cec21_f8_bias_shift,\
    cec21_f9_bias_shift, cec21_f10_bias_shift
# CEC 2021 - Bias & Shift & Rotation
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_bias_shift_rot, cec21_f2_bias_shift_rot,\
    cec21_f3_bias_shift_rot, cec21_f4_bias_shift_rot, cec21_f5_bias_shift_rot, cec21_f6_bias_shift_rot,\
    cec21_f7_bias_shift_rot, cec21_f8_bias_shift_rot, cec21_f9_bias_shift_rot, cec21_f10_bias_shift_rot
# CEC 2021 - Rotation
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_rot, cec21_f2_rot, cec21_f3_rot, cec21_f4_rot,\
    cec21_f5_rot, cec21_f6_rot, cec21_f7_rot, cec21_f8_rot, cec21_f9_rot, cec21_f10_rot
# CEC 2021 - Shift
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_shift, cec21_f2_shift, cec21_f3_shift, cec21_f4_shift,\
    cec21_f5_shift, cec21_f6_shift, cec21_f7_shift, cec21_f8_shift, cec21_f9_shift, cec21_f10_shift
# CEC 2021 - Shift & Rotation
from .definitions.CEC.TestSuite2021.cec2021 import cec21_f1_shift_rot, cec21_f2_shift_rot, cec21_f3_shift_rot,\
    cec21_f4_shift_rot, cec21_f5_shift_rot, cec21_f6_shift_rot, cec21_f7_shift_rot, cec21_f8_shift_rot,\
    cec21_f9_shift_rot, cec21_f10_shift_rot
# CEC 2019
from .definitions.CEC.TestSuite2019.cec2019 import cec19_f1, cec19_f2, cec19_f3, cec19_f4, cec19_f5, cec19_f6,\
    cec19_f7, cec19_f8, cec19_f9, cec19_f10"""
# CEC 2017
from .definitions.CEC.TestSuite2017.cec2017 import cec17_f1, cec17_f2, cec17_f3, cec17_f4, cec17_f5, cec17_f6,\
    cec17_f7, cec17_f8, cec17_f9, cec17_f10, cec17_f11, cec17_f12, cec17_f13, cec17_f14, cec17_f15, cec17_f16,\
    cec17_f17, cec17_f18, cec17_f19, cec17_f20, cec17_f21, cec17_f22, cec17_f23, cec17_f24, cec17_f25, cec17_f26,\
    cec17_f27, cec17_f28, cec17_f29, cec17_f30
# CEC 2014
"""from .definitions.CEC.TestSuite2014.cec2014 import cec14_f1, cec14_f2, cec14_f3, cec14_f4, cec14_f5, cec14_f6,\
    cec14_f7, cec14_f8, cec14_f9, cec14_f10, cec14_f11, cec14_f12, cec14_f13, cec14_f14, cec14_f15, cec14_f16,\
    cec14_f17, cec14_f18, cec14_f19, cec14_f20, cec14_f21, cec14_f22, cec14_f23, cec14_f24, cec14_f25, cec14_f26,\
    cec14_f27, cec14_f28, cec14_f29, cec14_f30
# CEC 2013
from .definitions.CEC.TestSuite2013.cec2013 import cec13_f1, cec13_f2, cec13_f3, cec13_f4, cec13_f5, cec13_f6,\
    cec13_f7, cec13_f8, cec13_f9, cec13_f10, cec13_f11, cec13_f12, cec13_f13, cec13_f14, cec13_f15, cec13_f16,\
    cec13_f17, cec13_f18, cec13_f19, cec13_f20, cec13_f21, cec13_f22, cec13_f23, cec13_f24, cec13_f25, cec13_f26,\
    cec13_f27, cec13_f28
# CEC 2005
import optproblems.cec2005 as opt05"""


functions = {
    # ---------------
    # --- STANDARD --
    # ---------------

    # N-D Test Functions A
    'Ackley': ackley,
    'Adjiman': adjiman,
    'Alpine01': alpine01,
    'Alpine02': alpine02,
    # N-D Test Functions B
    'Bohachevsky': bohachevsky,
    'Bukin06': bukin06,
    # N-D Test Functions C
    'Cross_in_tray': cross_in_tray,
    # N-D Test Functions D
    'Damavandi': damavandi,
    'Deceptive': deceptive,
    'DeVilliersGlasser02': de_villiers_glasser02,
    'Drop_wave': drop_wave,
    # N-D Test Functions E
    'EggHolder': egg_holder,
    # N-D Test Functions F
    'Ferretti': ferretti,
    # N-D Test Functions G
    'Griewank': griewank,
    # N-D Test Functions H
    'Holder_table': holder_table,
    # N-D Test Functions M
    'Michalewicz': michalewicz,
    'Mishra01': mishra01,
    # N-D Test Functions N
    'Nobile1': nobile1,
    'Nobile2': nobile2,
    'Nobile3': nobile3,
    # N-D Test Functions P
    'Plateau': plateau,
    # N-D Test Functions Q
    'Quintic': quintic,
    # N-D Test Functions R
    'Rastrigin': rastrigin,
    'Ripple01': ripple01,
    'Rosenbrock': rosenbrock,
    # N-D Test Functions S
    'Salomon': salomon,
    'Schwefel': schwefel,
    'Sphere': sphere,
    'Shubert': shubert,
    'Stochastic': stochastic,
    # N-D Test Functions V
    'Vincent': vincent,
    # N-D Test Functions W
    'Whitley': whitley,
    # N-D Test Functions X
    'XinSheYang02': xinsheyang02,
    'XinSheYang03': xinsheyang03,

    # ----------------
    # --- PERTURBED --
    # ----------------

    # N-D Perturbed Test Functions A
    'Ackley_shifted': ackley_shifted,
    'Ackley_shrinked': ackley_shrinked,
    'Alpine01_shrinked': alpine01_shrinked,
    'Alpine01_shifted': alpine01_shifted,
    # N-D Perturbed Test Functions R
    'Rosenbrock_shrinked': rosenbrock_shrinked,
    'Rosenbrock_shifted': rosenbrock_shifted,
    # N-D Perturbed Test Functions S
    'Sphere_shrinked': sphere_shrinked,
    'Sphere_shifted': sphere_shifted,
    # CEC 2017
    'CEC17-F1': cec17_f1,
    'CEC17-F2': cec17_f2,
    'CEC17-F3': cec17_f3,
    'CEC17-F4': cec17_f4,
    'CEC17-F5': cec17_f5,
    'CEC17-F6': cec17_f6,
    'CEC17-F7': cec17_f7,
    'CEC17-F8': cec17_f8,
    'CEC17-F9': cec17_f9,
    'CEC17-F10': cec17_f10,
    'CEC17-F11': cec17_f11,
    'CEC17-F12': cec17_f12,
    'CEC17-F13': cec17_f13,
    'CEC17-F14': cec17_f14,
    'CEC17-F15': cec17_f15,
    'CEC17-F16': cec17_f16,
    'CEC17-F17': cec17_f17,
    'CEC17-F18': cec17_f18,
    'CEC17-F19': cec17_f19,
    'CEC17-F20': cec17_f20,
    'CEC17-F21': cec17_f21,
    'CEC17-F22': cec17_f22,
    'CEC17-F23': cec17_f23,
    'CEC17-F24': cec17_f24,
    'CEC17-F25': cec17_f25,
    'CEC17-F26': cec17_f26,
    'CEC17-F27': cec17_f27,
    'CEC17-F28': cec17_f28,
    'CEC17-F29': cec17_f29,
    'CEC17-F30': cec17_f30,
    # CEC 2022
    'CEC22-F1': cec22_f1,
    'CEC22-F2': cec22_f2,
    'CEC22-F3': cec22_f3,
    'CEC22-F4': cec22_f4,
    'CEC22-F5': cec22_f5,
    'CEC22-F6': cec22_f6,
    'CEC22-F7': cec22_f7,
    'CEC22-F8': cec22_f8,
    'CEC22-F9': cec22_f9,
    'CEC22-F10': cec22_f10,
    'CEC22-F11': cec22_f11,
    'CEC22-F12': cec22_f12, }

""""# CEC 2021 - Basic
    'CEC21-F1-Basic': cec21_f1_basic,
    'CEC21-F2-Basic': cec21_f2_basic,
    'CEC21-F3-Basic': cec21_f3_basic,
    'CEC21-F4-Basic': cec21_f4_basic,
    'CEC21-F5-Basic': cec21_f5_basic,
    'CEC21-F6-Basic': cec21_f6_basic,
    'CEC21-F7-Basic': cec21_f7_basic,
    'CEC21-F8-Basic': cec21_f8_basic,
    'CEC21-F9-Basic': cec21_f9_basic,
    'CEC21-F10-Basic': cec21_f10_basic,

    # CEC 2021 - Bias
    'CEC21-F1-Bias': cec21_f1_bias,
    'CEC21-F2-Bias': cec21_f2_bias,
    'CEC21-F3-Bias': cec21_f3_bias,
    'CEC21-F4-Bias': cec21_f4_bias,
    'CEC21-F5-Bias': cec21_f5_bias,
    'CEC21-F6-Bias': cec21_f6_bias,
    'CEC21-F7-Bias': cec21_f7_bias,
    'CEC21-F8-Bias': cec21_f8_bias,
    'CEC21-F9-Bias': cec21_f9_bias,
    'CEC21-F10-Bias': cec21_f10_bias,

    # CEC 2021 - Bias & Rotation
    'CEC21-F1-Bias&Rotation': cec21_f1_bias_rot,
    'CEC21-F2-Bias&Rotation': cec21_f2_bias_rot,
    'CEC21-F3-Bias&Rotation': cec21_f3_bias_rot,
    'CEC21-F4-Bias&Rotation': cec21_f4_bias_rot,
    'CEC21-F5-Bias&Rotation': cec21_f5_bias_rot,
    'CEC21-F6-Bias&Rotation': cec21_f6_bias_rot,
    'CEC21-F7-Bias&Rotation': cec21_f7_bias_rot,
    'CEC21-F8-Bias&Rotation': cec21_f8_bias_rot,
    'CEC21-F9-Bias&Rotation': cec21_f9_bias_rot,
    'CEC21-F10-Bias&Rotation': cec21_f10_bias_rot,

    # CEC 2021 - Bias & Shift
    'CEC21-F1-Bias&Shift': cec21_f1_bias_shift,
    'CEC21-F2-Bias&Shift': cec21_f2_bias_shift,
    'CEC21-F3-Bias&Shift': cec21_f3_bias_shift,
    'CEC21-F4-Bias&Shift': cec21_f4_bias_shift,
    'CEC21-F5-Bias&Shift': cec21_f5_bias_shift,
    'CEC21-F6-Bias&Shift': cec21_f6_bias_shift,
    'CEC21-F7-Bias&Shift': cec21_f7_bias_shift,
    'CEC21-F8-Bias&Shift': cec21_f8_bias_shift,
    'CEC21-F9-Bias&Shift': cec21_f9_bias_shift,
    'CEC21-F10-Bias&Shift': cec21_f10_bias_shift,

    # CEC 2021 - Bias & Shift & Rotation
    'CEC21-F1-Bias&Shift&Rotation': cec21_f1_bias_shift_rot,
    'CEC21-F2-Bias&Shift&Rotation': cec21_f2_bias_shift_rot,
    'CEC21-F3-Bias&Shift&Rotation': cec21_f3_bias_shift_rot,
    'CEC21-F4-Bias&Shift&Rotation': cec21_f4_bias_shift_rot,
    'CEC21-F5-Bias&Shift&Rotation': cec21_f5_bias_shift_rot,
    'CEC21-F6-Bias&Shift&Rotation': cec21_f6_bias_shift_rot,
    'CEC21-F7-Bias&Shift&Rotation': cec21_f7_bias_shift_rot,
    'CEC21-F8-Bias&Shift&Rotation': cec21_f8_bias_shift_rot,
    'CEC21-F9-Bias&Shift&Rotation': cec21_f9_bias_shift_rot,
    'CEC21-F10-Bias&Shift&Rotation': cec21_f10_bias_shift_rot,

    # CEC 2021 - Rotation
    'CEC21-F1-Rotation': cec21_f1_rot,
    'CEC21-F2-Rotation': cec21_f2_rot,
    'CEC21-F3-Rotation': cec21_f3_rot,
    'CEC21-F4-Rotation': cec21_f4_rot,
    'CEC21-F5-Rotation': cec21_f5_rot,
    'CEC21-F6-Rotation': cec21_f6_rot,
    'CEC21-F7-Rotation': cec21_f7_rot,
    'CEC21-F8-Rotation': cec21_f8_rot,
    'CEC21-F9-Rotation': cec21_f9_rot,
    'CEC21-F10-Rotation': cec21_f10_rot,

    # CEC 2021 - Shift
    'CEC21-F1-Shift': cec21_f1_shift,
    'CEC21-F2-Shift': cec21_f2_shift,
    'CEC21-F3-Shift': cec21_f3_shift,
    'CEC21-F4-Shift': cec21_f4_shift,
    'CEC21-F5-Shift': cec21_f5_shift,
    'CEC21-F6-Shift': cec21_f6_shift,
    'CEC21-F7-Shift': cec21_f7_shift,
    'CEC21-F8-Shift': cec21_f8_shift,
    'CEC21-F9-Shift': cec21_f9_shift,
    'CEC21-F10-Shift': cec21_f10_shift,

    # CEC 2021 - Shift & Rotation
    'CEC21-F1-Shift&Rotation': cec21_f1_shift_rot,
    'CEC21-F2-Shift&Rotation': cec21_f2_shift_rot,
    'CEC21-F3-Shift&Rotation': cec21_f3_shift_rot,
    'CEC21-F4-Shift&Rotation': cec21_f4_shift_rot,
    'CEC21-F5-Shift&Rotation': cec21_f5_shift_rot,
    'CEC21-F6-Shift&Rotation': cec21_f6_shift_rot,
    'CEC21-F7-Shift&Rotation': cec21_f7_shift_rot,
    'CEC21-F8-Shift&Rotation': cec21_f8_shift_rot,
    'CEC21-F9-Shift&Rotation': cec21_f9_shift_rot,
    'CEC21-F10-Shift&Rotation': cec21_f10_shift_rot,

    # CEC 2019
    'CEC19-F1': cec19_f1,
    'CEC19-F2': cec19_f2,
    'CEC19-F3': cec19_f3,
    'CEC19-F4': cec19_f4,
    'CEC19-F5': cec19_f5,
    'CEC19-F6': cec19_f6,
    'CEC19-F7': cec19_f7,
    'CEC19-F8': cec19_f8,
    'CEC19-F9': cec19_f9,
    'CEC19-F10': cec19_f10,

    # CEC 2014
    'CEC14-F1': cec14_f1,
    'CEC14-F2': cec14_f2,
    'CEC14-F3': cec14_f3,
    'CEC14-F4': cec14_f4,
    'CEC14-F5': cec14_f5,
    'CEC14-F6': cec14_f6,
    'CEC14-F7': cec14_f7,
    'CEC14-F8': cec14_f8,
    'CEC14-F9': cec14_f9,
    'CEC14-F10': cec14_f10,
    'CEC14-F11': cec14_f11,
    'CEC14-F12': cec14_f12,
    'CEC14-F13': cec14_f13,
    'CEC14-F14': cec14_f14,
    'CEC14-F15': cec14_f15,
    'CEC14-F16': cec14_f16,
    'CEC14-F17': cec14_f17,
    'CEC14-F18': cec14_f18,
    'CEC14-F19': cec14_f19,
    'CEC14-F20': cec14_f20,
    'CEC14-F21': cec14_f21,
    'CEC14-F22': cec14_f22,
    'CEC14-F23': cec14_f23,
    'CEC14-F24': cec14_f24,
    'CEC14-F25': cec14_f25,
    'CEC14-F26': cec14_f26,
    'CEC14-F27': cec14_f27,
    'CEC14-F28': cec14_f28,
    'CEC14-F29': cec14_f29,
    'CEC14-F30': cec14_f30,

    # CEC 2013
    'CEC13-F1': cec13_f1,
    'CEC13-F2': cec13_f2,
    'CEC13-F3': cec13_f3,
    'CEC13-F4': cec13_f4,
    'CEC13-F5': cec13_f5,
    'CEC13-F6': cec13_f6,
    'CEC13-F7': cec13_f7,
    'CEC13-F8': cec13_f8,
    'CEC13-F9': cec13_f9,
    'CEC13-F10': cec13_f10,
    'CEC13-F11': cec13_f11,
    'CEC13-F12': cec13_f12,
    'CEC13-F13': cec13_f13,
    'CEC13-F14': cec13_f14,
    'CEC13-F15': cec13_f15,
    'CEC13-F16': cec13_f16,
    'CEC13-F17': cec13_f17,
    'CEC13-F18': cec13_f18,
    'CEC13-F19': cec13_f19,
    'CEC13-F20': cec13_f20,
    'CEC13-F21': cec13_f21,
    'CEC13-F22': cec13_f22,
    'CEC13-F23': cec13_f23,
    'CEC13-F24': cec13_f24,
    'CEC13-F25': cec13_f25,
    'CEC13-F26': cec13_f26,
    'CEC13-F27': cec13_f27,
    'CEC13-F28': cec13_f28,

    # CEC 2005
    'CEC05-F1': getattr(opt05, 'F1')(dimensions['CEC05-F1'][0]),
    'CEC05-F2': getattr(opt05, 'F2')(dimensions['CEC05-F2'][0]),
    'CEC05-F3': getattr(opt05, 'F3')(dimensions['CEC05-F3'][0]),
    'CEC05-F4': getattr(opt05, 'F4')(dimensions['CEC05-F4'][0]),
    'CEC05-F5': getattr(opt05, 'F5')(dimensions['CEC05-F5'][0]),
    'CEC05-F6': getattr(opt05, 'F6')(dimensions['CEC05-F6'][0]),
    'CEC05-F7': getattr(opt05, 'F7')(dimensions['CEC05-F7'][0]),
    'CEC05-F8': getattr(opt05, 'F8')(dimensions['CEC05-F8'][0]),
    'CEC05-F9': getattr(opt05, 'F9')(dimensions['CEC05-F9'][0]),
    'CEC05-F10': getattr(opt05, 'F10')(dimensions['CEC05-F10'][0]),
    'CEC05-F11': getattr(opt05, 'F11')(dimensions['CEC05-F11'][0]),
    'CEC05-F12': getattr(opt05, 'F12')(dimensions['CEC05-F12'][0]),
    'CEC05-F13': getattr(opt05, 'F13')(dimensions['CEC05-F13'][0]),
    'CEC05-F14': getattr(opt05, 'F14')(dimensions['CEC05-F14'][0]),
    'CEC05-F15': getattr(opt05, 'F15')(dimensions['CEC05-F15'][0]),
    'CEC05-F16': getattr(opt05, 'F16')(dimensions['CEC05-F16'][0]),
    'CEC05-F17': getattr(opt05, 'F17')(dimensions['CEC05-F17'][0]),
    'CEC05-F18': getattr(opt05, 'F18')(dimensions['CEC05-F18'][0]),
    'CEC05-F19': getattr(opt05, 'F19')(dimensions['CEC05-F19'][0]),
    'CEC05-F20': getattr(opt05, 'F20')(dimensions['CEC05-F20'][0]),
    'CEC05-F21': getattr(opt05, 'F21')(dimensions['CEC05-F21'][0]),
    'CEC05-F22': getattr(opt05, 'F22')(dimensions['CEC05-F22'][0]),
    'CEC05-F23': getattr(opt05, 'F23')(dimensions['CEC05-F23'][0]),
    'CEC05-F24': getattr(opt05, 'F24')(dimensions['CEC05-F24'][0]),
    'CEC05-F25': getattr(opt05, 'F25')(dimensions['CEC05-F25'][0]),}
"""
