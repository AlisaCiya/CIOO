import sys
import os
from . import nn

def newmodel(args):

    if args.name == "n":
        print("Creating new Yolo-n model")
        model = nn.yolo_v11_n(classes=args.nc_steps)
    elif args.name == "s":
        print("Creating new Yolo-s model")
        model = nn.yolo_v11_s(classes=args.nc_steps)
    elif args.name == "m":
        print("Creating new Yolo-m model")
        model = nn.yolo_v11_m(classes=args.nc_steps)
    elif args.name == "l":
        print("Creating new Yolo-l model")
        model = nn.yolo_v11_l(classes=args.nc_steps)
    elif args.name == "x":
        print("Creating new Yolo-x model")
        model = nn.yolo_v11_x(classes=args.nc_steps)
    else:
        raise ValueError(f"Unknown model name: Yolo-{args.name}")
    return model
    

def oldmodel(args):

    if args.name == "n":
        print("Creating old Yolo-n model")
        model = nn.yolo_v11_n(classes=args.nc_steps[:-1])
    elif args.name == "s":
        print("Creating old Yolo-s model")
        model = nn.yolo_v11_s(classes=args.nc_steps[:-1])
    elif args.name == "m":
        print("Creating old Yolo-m model")
        model = nn.yolo_v11_m(classes=args.nc_steps[:-1])
    elif args.name == "l":
        print("Creating old Yolo-l model")
        model = nn.yolo_v11_l(classes=args.nc_steps[:-1])
    elif args.name == "x":
        print("Creating old Yolo-x model")
        model = nn.yolo_v11_x(classes=args.nc_steps[:-1])
    else:
        raise ValueError(f"Unknown model name: Yolo-{args.name}")
    return model
    