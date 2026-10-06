#!/usr/bin/env python3
"""
NuCLS blinded HoVer-Net production inference v15.

One manifest row / FOV per SLURM task.

Hard boundaries:
- image-only inputs + external checkpoint only;
- no anchor boxes, reader labels, P-truth, scoring key, or NuCLS reference tables;
- L40S production runtime only;
- detection-only export; no invented instance confidence.
"""
from repo_paths import repo_root
import argparse, csv, hashlib, json, math, os, socket, tempfile, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from nucls_hovernet_architecture_v10 import HoVerNetOriginal

EXPECTED_CKPT_SHA = "5d1191d6bf72a077911aef12a75484ee3cab91ed81e7880c0361157094ff451d"
RUNTIME_NAME = "L40S_PYTORCH_2.10.0_CU126"
STATUS = "COMPLETE_BLINDED_GPU_PRODUCTION_V15"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def atomic_write_text(path, text):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def atomic_write_csv(path, fieldnames, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def remove_small(labels,min_size=10):
    sizes=np.bincount(labels.ravel())
    labels=labels.copy()
    labels[sizes[labels] < min_size]=0
    return labels

def postprocess_np_hv(pred):
    pred=np.asarray(pred,dtype=np.float32)
    blb=ndi.label((pred[...,0]>=0.5).astype(np.int32))[0]
    blb=remove_small(blb,10); blb[blb>0]=1

    h=cv2.normalize(pred[...,1],None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    v=cv2.normalize(pred[...,2],None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    sh=cv2.Sobel(h,cv2.CV_64F,1,0,ksize=21)
    sv=cv2.Sobel(v,cv2.CV_64F,0,1,ksize=21)
    sh=1-cv2.normalize(sh,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    sv=1-cv2.normalize(sv,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    overall=np.maximum(0,np.maximum(sh,sv)-(1-blb))
    dist=-cv2.GaussianBlur((1-overall)*blb,(3,3),0)

    marker=np.maximum(0,blb-(overall>=0.4).astype(np.int32))
    marker=ndi.binary_fill_holes(marker).astype(np.uint8)
    marker=cv2.morphologyEx(
        marker,cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    )
    marker=remove_small(ndi.label(marker)[0],10)
    return watershed(dist,markers=marker,mask=blb)

def prepare_image(image):
    oh,ow=image.shape[:2]
    w,h=round(ow*8/11),round(oh*8/11)
    resized=np.asarray(
        Image.fromarray(image).resize((w,h),resample=Image.Resampling.BICUBIC)
    )
    sy,sx=math.ceil(h/80),math.ceil(w/80)
    lh,lw=sy*80,sx*80
    padded=np.pad(
        resized,
        ((95,lh+270-h),(95,lw+270-w),(0,0)),
        mode="reflect"
    )
    return padded,(oh,ow),(h,w),(sy,sx),(lh,lw)

def run_model(model,padded,sy,sx,lh,lw,batch_size):
    raw=np.empty((lh,lw,4),np.float32)
    coords=[]; tiles=[]
    for iy in range(sy):
        for ix in range(sx):
            coords.append((iy,ix))
            t=padded[iy*80:iy*80+270,ix*80:ix*80+270].copy()
            tiles.append(torch.from_numpy(t).permute(2,0,1).float())

    t0=time.time()
    with torch.inference_mode():
        for k in range(0,len(tiles),batch_size):
            batch=torch.stack(tiles[k:k+batch_size]).cuda(non_blocking=False)
            out=model(batch)
            typ=out["tp"].argmax(1).detach().cpu().numpy().astype(np.float32)
            prob=out["np"].softmax(1)[:,1].detach().cpu().numpy()
            hv=out["hv"].permute(0,2,3,1).detach().cpu().numpy()
            for q,(iy,ix) in enumerate(coords[k:k+batch_size]):
                raw[iy*80:iy*80+80,ix*80:ix*80+80]=np.concatenate(
                    (typ[q,...,None],prob[q,...,None],hv[q]),axis=-1
                )
            del batch,out
    torch.cuda.synchronize()
    return raw,time.time()-t0,len(tiles)

def boxes_from_label_map(lab,ow,oh,w,h):
    sxback,syback=ow/w,oh/h
    rows=[]
    for j,L in enumerate([x for x in np.unique(lab) if x!=0],1):
        yy,xx=np.where(lab==L)
        rows.append({
            "instance_id":j,
            "xmin":float(xx.min()*sxback),
            "ymin":float(yy.min()*syback),
            "xmax":float((xx.max()+1)*sxback),
            "ymax":float((yy.max()+1)*syback),
        })
    return rows

def output_valid(csv_path, receipt_path, expected_fov, expected_image_sha):
    try:
        if not csv_path.exists() or not receipt_path.exists():
            return False
        rec=json.loads(receipt_path.read_text())
        if rec["status"]!=STATUS: return False
        if int(rec["fov_id"])!=expected_fov: return False
        if rec["image_sha256"]!=expected_image_sha: return False
        if rec["checkpoint_sha256"]!=EXPECTED_CKPT_SHA: return False
        if rec["production_runtime"]!=RUNTIME_NAME: return False
        if rec["scoring_key_opened"] is not False: return False
        if rec["reader_labels_opened"] is not False: return False
        if rec["reference_data_accessed"] is not False: return False
        if rec["output_csv_sha256"]!=sha256(csv_path): return False
        n=sum(1 for _ in csv.DictReader(csv_path.open()))
        return n==int(rec["num_instances"])
    except Exception:
        return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=str(repo_root()))
    ap.add_argument("--task-index",type=int,required=True)
    ap.add_argument("--batch-size",type=int,default=4)
    args=ap.parse_args()

    project=Path(args.project)
    manifest=list(csv.DictReader((project/"data/blinded/nucls_gpu_production_manifest_v15.csv").open()))
    if not (0 <= args.task_index < len(manifest)):
        raise SystemExit(f"Invalid task index {args.task_index}")
    r=manifest[args.task_index]
    fid=int(r["fov_id"])

    outdir=project/"outputs/blinded_inference_v15"
    recdir=project/"outputs/receipts_v15"
    outdir.mkdir(parents=True,exist_ok=True); recdir.mkdir(parents=True,exist_ok=True)
    out_csv=outdir/f"FOV{fid:02d}_instances.csv"
    receipt=recdir/f"FOV{fid:02d}_receipt.json"

    if output_valid(out_csv,receipt,fid,r["image_sha256"]):
        print(f"SKIP FOV{fid:02d}: validated production output already exists")
        return

    host=socket.gethostname()
    if host.startswith("__LOCAL_FORBIDDEN_HOST__"):
        raise RuntimeError(f"Forbidden incompatible node family: {host}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in L40S production job")
    if torch.cuda.device_count()!=1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
    gpu_name=torch.cuda.get_device_name(0)
    if "L40S" not in gpu_name.upper():
        raise RuntimeError(f"Production runtime requires L40S, got: {gpu_name}")

    image_path=project/"data/blinded"/r["image_filename"]
    ckpt=project/"checkpoints/hovernet_original_consep_type_tf2pytorch.tar"
    if sha256(ckpt)!=EXPECTED_CKPT_SHA:
        raise RuntimeError("Checkpoint SHA256 mismatch")
    if sha256(image_path)!=r["image_sha256"]:
        raise RuntimeError(f"Image SHA256 mismatch for FOV {fid}")

    image=np.asarray(Image.open(image_path).convert("RGB"))
    if image.shape[1]!=int(r["width_px"]) or image.shape[0]!=int(r["height_px"]):
        raise RuntimeError(f"Image geometry mismatch for FOV {fid}")

    torch.manual_seed(20260925+fid)
    torch.cuda.manual_seed_all(20260925+fid)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    weights=torch.load(ckpt,map_location="cpu",weights_only=True)["desc"]
    model=HoVerNetOriginal(5)
    model.load_state_dict(weights,strict=True)
    model.eval().cuda()

    padded,(oh,ow),(h,w),(sy,sx),(lh,lw)=prepare_image(image)
    torch.cuda.reset_peak_memory_stats()
    total_t0=time.time()
    raw,network_seconds,npatch=run_model(model,padded,sy,sx,lh,lw,args.batch_size)
    raw=raw[:h,:w]
    if not np.isfinite(raw).all():
        raise RuntimeError("Non-finite model output")

    lab=postprocess_np_hv(raw[...,1:])
    boxes=boxes_from_label_map(lab,ow,oh,w,h)

    fields=[
        "image_id","fov_id","prediction_id",
        "xmin_px","ymin_px","xmax_px","ymax_px",
        "detection_score","predicted_class",
        "image_sha256","checkpoint_sha256",
        "production_runtime","prediction_state",
    ]
    out_rows=[]
    for b in boxes:
        out_rows.append({
            "image_id":r["image_id"],
            "fov_id":fid,
            "prediction_id":f"FOV{fid:02d}_AI_{b['instance_id']:04d}",
            "xmin_px":b["xmin"],
            "ymin_px":b["ymin"],
            "xmax_px":b["xmax"],
            "ymax_px":b["ymax"],
            "detection_score":"NOT_AVAILABLE",
            "predicted_class":"NOT_SCORED_DETECTION_ONLY",
            "image_sha256":r["image_sha256"],
            "checkpoint_sha256":EXPECTED_CKPT_SHA,
            "production_runtime":RUNTIME_NAME,
            "prediction_state":STATUS,
        })

    atomic_write_csv(out_csv,fields,out_rows)
    total_seconds=time.time()-total_t0
    out_sha=sha256(out_csv)

    rec={
        "status":STATUS,
        "task_index":args.task_index,
        "fov_id":fid,
        "image_id":r["image_id"],
        "slide":r["slide"],
        "patient_id":r["patient_id"],
        "image_filename":r["image_filename"],
        "image_sha256":r["image_sha256"],
        "checkpoint_sha256":EXPECTED_CKPT_SHA,
        "production_runtime":RUNTIME_NAME,
        "preprocessing":"resize 8/11 using PIL bicubic; HoVer-Net original 270->80 tiling",
        "postprocessing":"official-source-derived np+hv watershed frozen before production",
        "batch_size":args.batch_size,
        "source_width_px":ow,
        "source_height_px":oh,
        "model_width_px":w,
        "model_height_px":h,
        "realized_x_scale":w/ow,
        "realized_y_scale":h/oh,
        "num_patches":npatch,
        "num_instances":len(boxes),
        "network_seconds":network_seconds,
        "total_inference_seconds":total_seconds,
        "gpu_peak_allocated_mb":torch.cuda.max_memory_allocated()/1024**2,
        "node":host,
        "gpu_name":gpu_name,
        "torch_version":torch.__version__,
        "torch_cuda":torch.version.cuda,
        "cudnn_version":torch.backends.cudnn.version(),
        "python_version":os.sys.version.replace("\n"," "),
        "output_csv":str(out_csv),
        "output_csv_sha256":out_sha,
        "detection_score_available":False,
        "class_scoring_authorized":False,
        "reference_data_accessed":False,
        "scoring_key_opened":False,
        "reader_labels_opened":False,
        "ai_accuracy_estimated":False,
    }
    atomic_write_text(receipt,json.dumps(rec,indent=2)+"\n")

    assert output_valid(out_csv,receipt,fid,r["image_sha256"])
    print(json.dumps(rec,indent=2))
    print(f"PRODUCTION_FOV_PASS FOV{fid:02d}")

if __name__=="__main__":
    main()
