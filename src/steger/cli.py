import argparse
import json
from pathlib import Path

from .comparison import compare_results
from .detector import lines_gauss
from .io import overlay,read_image,write_json
from .measurement import measure_bending


def main(argv=None):
    parser=argparse.ArgumentParser(description="Precision-oriented Steger extraction and validation")
    sub=parser.add_subparsers(dest="command",required=True)
    detect=sub.add_parser("detect")
    detect.add_argument("image");detect.add_argument("--mask")
    detect.add_argument("--sigma",type=float,default=1.5)
    detect.add_argument("--low",type=float,default=3);detect.add_argument("--high",type=float,default=8)
    detect.add_argument("--polarity",choices=["light","dark"],default="light")
    detect.add_argument("--model",choices=["none","bar-shaped","parabolic","gaussian"],default="bar-shaped")
    detect.add_argument("--no-width",action="store_true")
    detect.add_argument("--no-complete-junctions",action="store_true")
    detect.add_argument("--no-profile-refinement",action="store_true")
    detect.add_argument("--no-newton-refinement",action="store_true")
    detect.add_argument("--min-length",type=float,default=0)
    detect.add_argument("--output",required=True);detect.add_argument("--overlay")
    detect.add_argument("--measure-neck",action="store_true")
    detect.add_argument("--contour-index",type=int)
    compare=sub.add_parser("compare")
    compare.add_argument("candidate");compare.add_argument("reference");compare.add_argument("--output",required=True)
    compare.add_argument("--tolerance",type=float,default=.1);compare.add_argument("--p95-limit",type=float,default=.05)
    args=parser.parse_args(argv)
    try:
        if args.command=="compare":
            report=compare_results(json.loads(Path(args.candidate).read_text()),json.loads(Path(args.reference).read_text()),
                                   tolerance_px=args.tolerance,p95_limit_px=args.p95_limit)
            write_json(args.output,report)
            print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['passes_geometry_gate'] else 2
        image=read_image(args.image);mask=read_image(args.mask,mask=True) if args.mask else None
        result=lines_gauss(image,args.sigma,args.low,args.high,args.polarity,not args.no_width,args.model,
                           not args.no_complete_junctions,mask=mask,min_length=args.min_length,
                           refine=not args.no_newton_refinement,profile_refinement=not args.no_profile_refinement)
        data=result.to_dict();data['engine']='steger-precision'
        if args.measure_neck:
            if args.contour_index is None:
                if len(result.contours)!=1:
                    raise ValueError("Neck measurement requires exactly one contour or an explicit --contour-index; inspect overlay first")
                c=result.contours[0]
            else:c=result.contours[args.contour_index]
            data['neck_measurement']=measure_bending(c).to_dict()
        write_json(args.output,data)
        if args.overlay:overlay(image,result,args.overlay)
        print(f"{len(result.contours)} contours, {result.diagnostics['output_points']} points -> {args.output}")
        print("HALCON equivalence is unverified until tested against a licensed reference installation.")
        return 0
    except (ValueError,IndexError,OSError) as e:
        parser.error(str(e))


if __name__=='__main__':raise SystemExit(main())
