"""Run on a licensed HALCON Windows/Linux installation, using the SAME input.

Example: python tools/export_halcon.py input.tiff --output reference.json
This script intentionally imports no steger code so it can be copied standalone.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('image');p.add_argument('--output',required=True)
    p.add_argument('--sigma',type=float,default=1.5);p.add_argument('--low',type=float,default=3)
    p.add_argument('--high',type=float,default=8);p.add_argument('--polarity',default='light',choices=['light','dark'])
    p.add_argument('--model',default='bar-shaped',choices=['none','bar-shaped','gaussian','parabolic'])
    p.add_argument('--no-width',action='store_true');p.add_argument('--no-complete-junctions',action='store_true')
    p.add_argument('--disable-simd',action='store_true')
    a=p.parse_args()
    try:import halcon as ha
    except ImportError:raise SystemExit('Install matching mvtec-halcon bindings AND licensed native HALCON first.')
    disabled=[]
    if a.disable_simd:
        for key in ['sse2_enable','avx2_enable','avx_enable','avx512f_enable']:
            try:ha.set_system(key,'false');disabled.append(key)
            except Exception:pass # Availability is version/platform dependent, recorded below.
    image=ha.read_image(str(Path(a.image).resolve()))
    contours=ha.lines_gauss(image,a.sigma,a.low,a.high,a.polarity,'false' if a.no_width else 'true',
                            a.model,'false' if a.no_complete_junctions else 'true')
    names=['angle','response']
    if not a.no_width:names+=['width_left','width_right']
    if not a.no_width and a.model!='none':names+=['asymmetry','contrast']
    items=[]
    for index in range(1,int(ha.count_obj(contours))+1):
        c=ha.select_obj(contours,index)
        row,col=ha.get_contour_xld(c)
        attrs={key:list(ha.get_contour_attrib_xld(c,key)) for key in names}
        items.append({'row':list(row),'col':list(col),'attributes':attrs})
    data={'schema_version':1,'engine':'HALCON','version':list(ha.get_system('version')),
          'source_image':str(Path(a.image).resolve()),'image_sha256':hashlib.sha256(Path(a.image).read_bytes()).hexdigest(),
          'parameters':{'sigma':a.sigma,'low':a.low,'high':a.high,'light_dark':a.polarity,
                        'line_model':a.model,'extract_width':not a.no_width,
                        'complete_junctions':not a.no_complete_junctions},
          'disabled_simd':disabled,'contours':items}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
