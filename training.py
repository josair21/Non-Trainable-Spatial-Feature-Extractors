"""Self-contained data loading and training used by paper/train.py."""

import json, random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from models.minirocketbased import RocketExtractor, model as MiniRocketHybrid
from models.visualprimitives import model as VisualPrimitives

ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"; RESULTS=ROOT/"results"
MEAN=(.485,.456,.406); STD=(.229,.224,.225)

def _split(labels,fraction=.1,seed=42):
 labels=torch.as_tensor(labels); g=torch.Generator().manual_seed(seed); train=[]; val=[]
 for cls in labels.unique(sorted=True):
  ids=(labels==cls).nonzero().flatten(); order=torch.randperm(len(ids),generator=g)
  n=max(1,round(len(ids)*fraction)); val+=ids[order[:n]].tolist(); train+=ids[order[n:]].tolist()
 return train,val

def _loaders(dataset,batch,download):
 test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
 if dataset=="stl10":
  train_tf=transforms.Compose([transforms.RandomCrop(96,padding=12,padding_mode="reflect"),
      transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
  source=datasets.STL10(DATA,split="train",download=download,transform=train_tf)
  val_source=datasets.STL10(DATA,split="train",download=False,transform=test_tf)
  test=datasets.STL10(DATA,split="test",download=False,transform=test_tf)
  train_ids,val_ids=_split(source.labels)
 elif dataset=="imagenette":
  train_tf=transforms.Compose([transforms.RandomResizedCrop(160),transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
  final_tf=transforms.Compose([transforms.Resize(176),transforms.CenterCrop(160),
      transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
  source=datasets.Imagenette(DATA,split="train",size="160px",download=download,transform=train_tf)
  val_source=datasets.Imagenette(DATA,split="train",size="160px",download=False,transform=final_tf)
  test=datasets.Imagenette(DATA,split="val",size="160px",download=False,transform=final_tf)
  train_ids,val_ids=_split([label for _,label in source._samples])
 else:
  train_tf=transforms.Compose([transforms.RandomCrop(64,padding=6,padding_mode="reflect"),
      transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor(),
      transforms.Normalize(MEAN,STD)])
  source=datasets.EuroSAT(DATA,download=download,transform=train_tf)
  val_source=datasets.EuroSAT(DATA,download=False,transform=test_tf)
  test_source=datasets.EuroSAT(DATA,download=False,transform=test_tf)
  labels=torch.as_tensor(source.targets); g=torch.Generator().manual_seed(42)
  train_ids=[]; val_ids=[]; test_ids=[]
  for cls in labels.unique(sorted=True):
   ids=(labels==cls).nonzero().flatten(); order=torch.randperm(len(ids),generator=g)
   nt=round(.2*len(ids)); remain=ids[order[nt:]]; nv=round(.1*len(remain))
   test_ids+=ids[order[:nt]].tolist(); val_ids+=remain[:nv].tolist(); train_ids+=remain[nv:].tolist()
  test=Subset(test_source,test_ids)
 options=dict(num_workers=0,pin_memory=True)
 return (DataLoader(Subset(source,train_ids),batch_size=batch,shuffle=True,**options),
         DataLoader(Subset(val_source,val_ids),batch_size=batch,shuffle=False,**options),
         DataLoader(test,batch_size=batch,shuffle=False,**options))

def run(model_name,dataset,download=False):
 settings={"stl10":((96,96),384,8,("identity","hflip")),
           "eurosat":((64,64),512,4,("identity","hflip","vflip","hvflip")),
           "imagenette":((160,160),256,4,("identity","hflip"))}
 shape,batch,views,tta=settings[dataset]; seed=42
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 output=RESULTS/model_name/dataset; output.mkdir(parents=True,exist_ok=True)
 log_path=output/"training.log"; log_path.write_text("")
 def log(message):
  line=f"{datetime.now().isoformat(timespec='seconds')} {message}"; print(line,flush=True)
  with log_path.open("a") as stream: stream.write(line+"\n")
 train,val,test=_loaders(dataset,batch,download); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 if model_name=="visualprimitives": extractor=VisualPrimitives(3,shape,24000)
 elif dataset=="stl10": extractor=RocketExtractor(3,shape,15000)
 else: extractor=MiniRocketHybrid(3,shape,15000)
 extractor=extractor.to(device); images,_=next(iter(train)); extractor.fit(images.to(device))
 trainable=sum(p.numel() for p in extractor.parameters() if p.requires_grad)
 if trainable: raise RuntimeError(f"extractor has {trainable} trainable parameters")
 log(f"train={len(train.dataset)} validation={len(val.dataset)} test={len(test.dataset)} features={extractor.total_features} trainable=0")
 def extract(loader,variant="identity"):
  xs=[]; ys=[]; extractor.eval()
  with torch.inference_mode():
   for image,label in loader:
    image=image.to(device)
    if variant in ("hflip","hvflip"): image=torch.flip(image,(-1,))
    if variant in ("vflip","hvflip"): image=torch.flip(image,(-2,))
    xs.append(extractor(image).half().cpu()); ys.append(label)
  return torch.cat(xs),torch.cat(ys)
 xs=[]; ys=[]
 for view in range(views):
  x,y=extract(train); xs.append(x); ys.append(y); log(f"cached_train_view={view+1}")
 x_train=torch.cat(xs); y_train=torch.cat(ys); del xs,ys
 mean=x_train.float().mean(0); std=x_train.float().std(0).clamp_min(1e-4)
 x_val,y_val=extract(val); mean_d,std_d=mean.to(device),std.to(device)
 cache=DataLoader(TensorDataset(x_train,y_train),batch_size=1024,shuffle=True,num_workers=0,pin_memory=True)
 def logits(classifier,features):
  result=[]; classifier.eval()
  with torch.inference_mode():
   for start in range(0,len(features),1024):
    z=features[start:start+1024].to(device).float(); result.append(classifier((z-mean_d)/std_d).cpu())
  return torch.cat(result)
 criterion=torch.nn.CrossEntropyLoss(label_smoothing=.02); best=-1; selected=None
 for wd in (.1,.5,1.,2.):
  classifier=torch.nn.Linear(extractor.total_features,10).to(device)
  optimizer=torch.optim.AdamW(classifier.parameters(),lr=7e-4,weight_decay=wd)
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60,eta_min=1e-6)
  for epoch in range(1,61):
   classifier.train()
   for features,target in cache:
    features=features.to(device).float(); target=target.to(device)
    features=torch.nn.functional.dropout((features-mean_d)/std_d,p=.04,training=True)
    optimizer.zero_grad(set_to_none=True); loss=criterion(classifier(features),target)
    loss.backward(); optimizer.step()
   scheduler.step()
   if epoch==1 or epoch%5==0:
    accuracy=(logits(classifier,x_val).argmax(1)==y_val).float().mean().item()
    log(f"weight_decay={wd:g} epoch={epoch} validation_accuracy={accuracy:.4f}")
    if accuracy>best: best=accuracy; selected=(classifier.state_dict(),epoch,wd)
 classifier=torch.nn.Linear(extractor.total_features,10).to(device); classifier.load_state_dict(selected[0])
 ensemble=None; single=None; y_test=None
 for variant in tta:
  features,y_test=extract(test,variant); current=logits(classifier,features)
  if variant=="identity": single=current
  ensemble=current if ensemble is None else ensemble+current
 metrics={"validation_accuracy":best,"test_single_accuracy":(single.argmax(1)==y_test).float().mean().item(),
          "test_accuracy":(ensemble.argmax(1)==y_test).float().mean().item(),"best_epoch":selected[1],
          "best_weight_decay":selected[2],"extractor_features":extractor.total_features,"extractor_trainable":0}
 config={"model":model_name,"dataset":dataset,"seed":42,"shape":shape,"views":views,"tta":tta}
 torch.save({"model":model_name,"dataset":dataset,"config":config,"metrics":metrics,
             "extractor":extractor.state_dict(),"classifier":selected[0],"scaler":{"mean":mean,"std":std}},
            output/"checkpoint.pt")
 (output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
 log(f"finished validation={best:.4f} test={metrics['test_accuracy']:.4f}")
