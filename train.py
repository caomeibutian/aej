import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11s-our.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    #model.train(data='C:/Users/Win/Desktop/yolo11-main/DUT-UAV/data.yaml',
    #model.train(data='C:/Users/Win/Desktop/yolo11-main/bird_uav/data.yaml',
    model.train(data='C:/Users/Win/Desktop/yolo11-main/hongwai/data.yaml',
    #model.train(data='C:/Users/Win/Desktop/yolo11-main/hongwai/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
