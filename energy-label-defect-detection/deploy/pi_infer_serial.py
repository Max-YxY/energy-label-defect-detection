#!/usr/bin/env python3
"""树莓派能效标签检测 + 串口发送 — 无box版 + 时间平滑"""
import cv2, numpy as np, onnxruntime as ort, serial, serial.tools.list_ports
import time, os, sys

CLASS_NAMES={0:'level_1',1:'level_2',2:'level_3',3:'level_4',4:'level_5',5:'stain',6:'damage',7:'wrinkle',8:'label'}
NC=9; LABEL_ID=8; ENERGY_LEVEL_IDS={0,1,2,3,4}; DEFECT_IDS={5,6,7}
MAIN_INPUT_SIZE=320; CONF_THRESHOLD=0.25; IOU_THRESHOLD=0.45
TARGET_FPS=6; SERIAL_PORT=None; SERIAL_BAUD=115200
MODEL_DIR="/home/pi/yolo_model"; MAIN_MODEL=os.path.join(MODEL_DIR,"best.onnx")
SHOW_DISPLAY='--display' in sys.argv or '-d' in sys.argv
SMOOTH_FRAMES=3  # 无检测时保持显示帧数

def find_serial():
    for p in serial.tools.list_ports.comports():
        if "CH340" in p.description or "USB" in p.description: return p.device
    return None

def build_cmd(el,defects,_=False):
    lv=f"L{el}" if el else "L?"
    ds="+".join(d.upper() for d in defects) if defects else "OK"
    return f"{lv},{ds},OK"

def send_result(ser,el,defects,_=False):
    try: ser.write(f"{build_cmd(el,defects)}\r\n".encode())
    except: pass

def load_model(path):
    if not os.path.exists(path): return None,None
    so=ort.SessionOptions(); so.enable_cpu_mem_arena=True
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s=ort.InferenceSession(path,sess_options=so,providers=['CPUExecutionProvider'])
    print(f"  ok {os.path.basename(path)} ({s.get_inputs()[0].shape})")
    return s,s.get_inputs()[0].name

def nms(boxes,scores,iou):
    if len(boxes)==0: return[]
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0.,xx2-xx1)*np.maximum(0.,yy2-yy1)
        ovr=inter/(areas[i]+areas[order[1:]]-inter+1e-10)
        order=order[(ovr<=iou).nonzero()[0]+1]
    return keep

def postprocess(out,shape,conf,iou):
    p=np.squeeze(out).T; bc,sc=p[:,:4],p[:,4:]
    ms=np.max(sc,1); ci=np.argmax(sc,1)
    v=ms>=conf
    if not np.any(v): return[],[],[]
    bc=bc[v]; ms=ms[v]; ci=ci[v]
    xc,yc,w,h=bc[:,0],bc[:,1],bc[:,2],bc[:,3]
    b=np.stack([np.clip(xc-w/2,0,1),np.clip(yc-h/2,0,1),np.clip(xc+w/2,0,1),np.clip(yc+h/2,0,1)],1)
    k=nms(b,ms,iou)
    if not k: return[],[],[]
    fb=b[k]; fs=ms[k]; fc=ci[k]
    hi,wi=shape[:2]; fb[:,[0,2]]*=wi; fb[:,[1,3]]*=hi
    return fb.astype(np.int32),fs,fc

def draw_display(frame,all_boxes,energy_level,defects,_,fps=0):
    img=frame.copy(); h,w=img.shape[:2]
    for b in all_boxes:
        x1,y1,x2,y2=[int(v) for v in b['bbox']]; cid=b['class_id']
        if cid not in DEFECT_IDS: continue
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),4)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),1)
        lb=f"{b['class_name']} {b['confidence']:.2f}"
        (tw,th),_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.rectangle(img,(x1,y1-th-4),(x1+tw+4,y1),(0,0,255),-1)
        cv2.putText(img,lb,(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    for b in all_boxes:
        x1,y1,x2,y2=[int(v) for v in b['bbox']]; cid=b['class_id']
        if cid in DEFECT_IDS: continue
        color=(0,200,0) if cid in ENERGY_LEVEL_IDS else (200,200,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        lb=f"{b['class_name']} {b['confidence']:.2f}"
        (tw,th),_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)
        cv2.rectangle(img,(x1,y1-th-4),(x1+tw+4,y1),color,-1)
        cv2.putText(img,lb,(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    ov=img.copy(); cv2.rectangle(ov,(0,0),(w,60+len(defects)*22),(0,0,0),-1); cv2.addWeighted(ov,0.3,img,0.7,0,img)
    lv=str(energy_level) if energy_level else '?'
    cv2.putText(img,f"Level:{lv}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
    cv2.putText(img,f"{fps:.0f}FPS",(w-75,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    y0=45
    if defects:
        for i,d in enumerate(defects):
            y=y0+i*22
            (tw,th),_=cv2.getTextSize(d['defect_type'],cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
            ov2=img.copy(); cv2.rectangle(ov2,(8,y-th-2),(12+tw,y+2),(0,0,200),-1); cv2.addWeighted(ov2,0.5,img,0.5,0,img)
            cv2.putText(img,d['defect_type'],(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
    else: cv2.putText(img,"No Defects",(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),1)
    return img

def main():
    prev_boxes=[]; prev_el=None; prev_defects=[]; no_detect=0
    print("="*50); print("  能效标签检测 - 无box版")
    if SHOW_DISPLAY: print("  屏幕显示: ON"); print("  时间平滑: ON")
    print("="*50)
    print("\n加载模型...")
    sess,iname=load_model(MAIN_MODEL)
    if sess is None: sys.exit(1)
    port=SERIAL_PORT or find_serial(); ser=None
    if port:
        try: ser=serial.Serial(port,SERIAL_BAUD,timeout=0.1); print(f"\n串口: {port} ok")
        except: print(f"\n串口不可用")
    else: print("\n未发现串口")
    print("\n启动摄像头...")
    cam=None; picam2=None; cap=None
    try:
        from picamera2 import Picamera2
        picam2=Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size":(640,480)}))
        picam2.set_controls({"AeEnable": False, "ExposureTime": 12000, "AnalogueGain": 1.0, "AwbEnable": True, "Contrast": 1.3})
        picam2.start(); cam="CSI"; print("  CSI 摄像头 ok")
    except Exception as e:
        print(f"  CSI失败, USB..."); cap=cv2.VideoCapture(0)
        if not cap.isOpened(): print("  无法打开摄像头"); sys.exit(1); cam="USB"; print("  USB 摄像头 ok")
    if SHOW_DISPLAY:
        cv2.namedWindow("Energy Label Detection",cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Energy Label Detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    fc=0; ti=1.0/TARGET_FPS
    print(f"\n检测中, Ctrl+C停止"); print("-"*50)
    try:
        while True:
            t0=time.perf_counter()
            if cam=="CSI":
                f=picam2.capture_array("main"); f=cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
            else:
                ret,f=cap.read()
                if not ret: break
            oh,ow=f.shape[:2]
            ii=cv2.resize(f,(MAIN_INPUT_SIZE,MAIN_INPUT_SIZE))
            ten=np.expand_dims(np.transpose(ii.astype(np.float32)/255.,(2,0,1)),0)
            outs=sess.run(None,{iname:ten}); ims=(time.perf_counter()-t0)*1000
            boxes,scores,cids=postprocess(outs[0],(oh,ow),CONF_THRESHOLD,IOU_THRESHOLD)
            all_boxes=[]
            for box,score,cid in zip(boxes,scores,cids):
                all_boxes.append({'class_id':int(cid),'class_name':CLASS_NAMES.get(int(cid),'?'),'confidence':float(score),'bbox':[float(box[0]),float(box[1]),float(box[2]),float(box[3])]})
            el=None
            eb=[b for b in all_boxes if b['class_id'] in ENERGY_LEVEL_IDS]
            if eb: eb.sort(key=lambda x:x['confidence'],reverse=True); el=eb[0]['class_id']+1
            seen=set(); defects=[]
            for b in sorted([b for b in all_boxes if b['class_id'] in DEFECT_IDS],key=lambda x:x['confidence'],reverse=True):
                if b['class_id'] not in seen: defects.append({'defect_type':b['class_name'],'confidence':b['confidence']}); seen.add(b['class_id'])
            
            # 时间平滑: 当前帧有检测则更新历史，无检测则沿用
            if len(all_boxes)>0:
                prev_boxes=all_boxes; prev_el=el; prev_defects=defects; no_detect=0
            else:
                no_detect+=1
                if no_detect<=SMOOTH_FRAMES and len(prev_boxes)>0:
                    all_boxes=prev_boxes; el=prev_el; defects=prev_defects
            
            fc+=1
            if ser: send_result(ser,el,[d['defect_type'] for d in defects])
            ls=str(el) if el else '?'; ds="+".join(d['defect_type'] for d in defects) if defects else "-"
            print(f"[{fc:>4}] Lv{ls}  {ds:>20}  {ims:.0f}ms")
            if SHOW_DISPLAY:
                fps_val=1000./ims if ims>0 else 0
                di=draw_display(f,all_boxes,el,defects,None,fps_val)
                cv2.imshow("Energy Label Detection",di)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            sleep_ms=(ti*1000)-(time.perf_counter()-t0)*1000
            if sleep_ms>0: time.sleep(sleep_ms/1000.)
    except KeyboardInterrupt: print("\n停止")
    except Exception as e: print(f"\n错误: {e}")
    finally:
        if cam=="CSI": picam2.stop(); picam2.close()
        else: cap.release()
        if SHOW_DISPLAY: cv2.destroyAllWindows()
        if ser: ser.close(); print("已退出")
if __name__=="__main__": main()
