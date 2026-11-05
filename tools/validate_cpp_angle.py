"""Compile an independent literal C++ oracle and compare float bits to Python.

Requires MSVC on Windows. The oracle reproduces OneSperm.cpp:887-903, not
HALCON's line extraction. No HALCON installation or license is used here.
"""
import ctypes
import json
from pathlib import Path

import numpy as np
from setuptools._distutils.ccompiler import new_compiler

from steger import measure_direction_angles


SOURCE = r'''
#include <vector>
#include <algorithm>
#include <limits>
extern "C" __declspec(dllexport)
int cpp_smooth(const double* angles, int size, float* output) {
    std::vector<float> m_neckAngles;
    for (int i=0; i<size; ++i) m_neckAngles.push_back((float)angles[i]);
    int filterSize=10, vecSize=(int)m_neckAngles.size();
    int count=0;
    for (int kk=0; kk<vecSize-filterSize; ++kk) {
        float angleSum=0.;
        for (int jj=0; jj<filterSize; ++jj) angleSum += m_neckAngles[kk+jj];
        output[count++]=angleSum/(float)filterSize;
    }
    return count;
}
extern "C" __declspec(dllexport)
float cpp_bend(const double* angles, int size) {
    std::vector<float> filtered(size>10 ? size-10 : 0);
    int count=cpp_smooth(angles,size,filtered.data());
    if(!count) return std::numeric_limits<float>::quiet_NaN();
    auto lo=std::min_element(filtered.begin(),filtered.end());
    auto hi=std::max_element(filtered.begin(),filtered.end());
    return (float)((*hi-*lo)*180.0/3.1415926535897932384626433832795);
}
'''


def main():
    root=Path('reports/neck-analysis/cpp-angle-20261105/oracle').resolve()
    root.mkdir(parents=True,exist_ok=True)
    source=root/'cpp_angle.cpp'
    source.write_text(SOURCE,encoding='utf-8')
    compiler=new_compiler(compiler='msvc')
    objects=compiler.compile([str(source)],output_dir=str(root),extra_postargs=['/O2','/fp:strict'])
    dll=root/'cpp_angle.dll'
    compiler.link_shared_object(objects,str(dll))
    lib=ctypes.CDLL(str(dll))
    double_ptr=ctypes.POINTER(ctypes.c_double)
    float_ptr=ctypes.POINTER(ctypes.c_float)
    lib.cpp_bend.argtypes=[double_ptr,ctypes.c_int]
    lib.cpp_bend.restype=ctypes.c_float
    lib.cpp_smooth.argtypes=[double_ptr,ctypes.c_int,float_ptr]
    lib.cpp_smooth.restype=ctypes.c_int
    rng=np.random.default_rng(20261105)
    checked=0
    cases=[rng.uniform(0,2*np.pi,n) for n in (11,12,20,31,64,101,999) for _ in range(100)]
    cases += [np.zeros(11),np.r_[np.zeros(11),6.],np.r_[np.zeros(20),np.full(20,2*np.pi-.01)]]
    for angles in cases:
        a=np.asarray(angles,dtype=np.float64)
        output=np.empty(len(a)-10,dtype=np.float32)
        count=lib.cpp_smooth(a.ctypes.data_as(double_ptr),len(a),output.ctypes.data_as(float_ptr))
        expected=lib.cpp_bend(a.ctypes.data_as(double_ptr),len(a))
        actual=measure_direction_angles(a)
        assert count==actual.window_count
        assert output.tobytes()==np.asarray(actual.smoothed_angles_rad,dtype=np.float32).tobytes()
        assert np.float32(expected).tobytes()==np.float32(actual.angle_deg).tobytes()
        checked+=1
    report=dict(cases=checked,bitwise_exact=True,compiler='MSVC /O2 /fp:strict',
                source='OneSperm.cpp:887-903 arithmetic and bounds',
                limitation='Checks angle computation on equal angle arrays, not Python/HALCON detector parity')
    (root.parent/'cpp-oracle-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
